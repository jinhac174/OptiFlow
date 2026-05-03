from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.rng = np.random.RandomState(0)
        self._prestacked = False  # Whether frame stacking has already been materialized in the dataset.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def set_seed(self, seed: int):
        """Set the dataset-local RNG used for batch sampling and augmentation."""
        self.rng = np.random.RandomState(int(seed))

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return self.rng.randint(self.size, size=num_idxs)

    def _prestack_frames(self):
        """Materialize frame stacks once to avoid repeated batch-time concatenation."""
        if self.frame_stack is None or self._prestacked:
            return

        idxs = np.arange(self.size)
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        observations = self['observations']
        next_observations = self['next_observations']

        obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
        next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
        for i in reversed(range(self.frame_stack)):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], observations))
            if i != self.frame_stack - 1:
                next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], observations))
        next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], next_observations))

        stacked_obs = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
        stacked_next_obs = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), stacked_obs)
        jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), stacked_next_obs)
        self._dict['observations'] = stacked_obs
        self._dict['next_observations'] = stacked_next_obs
        self._prestacked = True

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if self.frame_stack is not None and not self._prestacked:
            self._prestack_frames()
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None and not self._prestacked:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if self.rng.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        result['indices'] = np.asarray(idxs, dtype=np.int32)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = self.rng.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


