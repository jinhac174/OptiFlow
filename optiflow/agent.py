"""OptiFlow agent (paper Sec. 4)."""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax

from .encoders import encoder_modules
from .flax_utils import ModuleDict, TrainState, nonpytree_field
from .networks import FlowPolicy, NNPolicy, Value


class OptiFlowAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ---- Critic aggregation -----------------------------------------------

    def _critic_agg(self, observations, actions, params, is_encoded=False):
        qs = self.network.select('critic')(observations, actions=actions, params=params, is_encoded=is_encoded)
        if qs.ndim == 2:
            return qs.min(axis=0) if self.config['q_agg'] == 'min' else qs.mean(axis=0)
        return qs

    def _critic_with_std(self, observations, actions, params):
        qs = self.network.select('critic')(observations, actions=actions, params=params)
        if qs.ndim == 2:
            agg = qs.min(axis=0) if self.config['q_agg'] == 'min' else qs.mean(axis=0)
            std = qs.std(axis=0)
            return agg, std
        return qs, jnp.zeros_like(qs)

    # ---- Action sampling --------------------------------------------------

    def _sample_from_onestep(self, observations, noises, params=None):
        if params is None:
            params = self.network.params
        return jnp.clip(
            self.network.select('actor_onestep')(observations, noises, params=params),
            -1, 1,
        )

    def _sample_from_flow(self, observations, noises, params=None):
        if params is None:
            params = self.network.params
        actions = noises
        for i in range(self.config['flow_steps']):
            t_shape = (*actions.shape[:-1], 1)
            t = jnp.full(t_shape, i / self.config['flow_steps'])
            vels = self.network.select('actor_bc')(
                observations, actions, t, params=params,
            )
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    # ---- Critic loss (TD) -------------------------------------------------

    def critic_loss(self, batch, grad_params, rng):
        rng, student_rng, teacher_rng = jax.random.split(rng, 3)
        next_obs = batch['next_observations']
        B = next_obs.shape[0]
        act_shape = (B, self.config['action_dim'])

        def aggregate_next_q(actions):
            actions = jax.lax.stop_gradient(jnp.clip(actions, -1, 1))
            next_qs = self.network.select('target_critic')(
                next_obs, actions=actions
            )
            return next_qs.min(axis=0) if self.config['q_agg'] == 'min' else next_qs.mean(axis=0)

        next_q_student = aggregate_next_q(
            self._sample_from_onestep(
                next_obs, jax.random.normal(student_rng, act_shape),
            )
        )
        if self.config['use_vabc_td_target']:
            next_q_teacher = aggregate_next_q(
                self._sample_from_flow(
                    next_obs, jax.random.normal(teacher_rng, act_shape),
                )
            )
            next_q = 0.5 * (next_q_student + next_q_teacher)
        else:
            next_q_teacher = next_q_student
            next_q = next_q_student
        target_q = jax.lax.stop_gradient(
            batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        )
        q = self.network.select('critic')(
            batch['observations'], actions=batch['actions'], params=grad_params
        )
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss':    critic_loss,
            'q_mean':         q.mean(),
            'q_max':          q.max(),
            'q_min':          q.min(),
            'next_q_student_mean': next_q_student.mean(),
            'next_q_teacher_mean': next_q_teacher.mean(),
            'next_q_mean':    next_q.mean(),
            'target_q_mean':  target_q.mean(),
        }

    # ---- Value-aware reference flow (VaBC) --------------------------------

    def _critic_eval_target(self, observations, actions):
        qs = self.network.select('target_critic')(observations, actions)
        if qs.ndim == 2:
            return qs.min(axis=0) if self.config['q_agg'] == 'min' else qs.mean(axis=0)
        return qs

    def _vabc_guidance(self, q_data, q_ref, q_scale):
        # g_eta = softmax([Q(s,a), Q(s, mu_theta(s,z))] / eta), select dataset slot.
        scale = q_scale / self.config['eta_temperature']
        logits = jnp.stack([scale * q_data, scale * q_ref], axis=-1)
        return jax.nn.softmax(logits, axis=-1)[..., 0]

    def vabc_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, ref_rng = jax.random.split(rng, 4)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t   = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc')(
            batch['observations'], x_t, t, params=grad_params,
        )
        flow_error = jnp.mean((pred - vel) ** 2, axis=1)

        ref_noise = jax.random.normal(ref_rng, (batch_size, action_dim))
        ref_actions = jnp.clip(
            self.one_step_actions(
                batch['observations'], ref_noise, grad_params=grad_params,
            ),
            -1, 1,
        )
        ref_actions = jax.lax.stop_gradient(ref_actions)

        q_data = self._critic_eval_target(batch['observations'], batch['actions'])
        q_ref  = self._critic_eval_target(batch['observations'], ref_actions)
        q_scale = jax.lax.stop_gradient(1.0 / (jnp.mean(jnp.abs(q_data)) + 1e-6))
        guidance = jax.lax.stop_gradient(
            self._vabc_guidance(q_data, q_ref, q_scale)
        )

        loss = jnp.mean(guidance * flow_error)
        info = {
            'guidance_mean': guidance.mean(),
            'guidance_max':  guidance.max(),
            'guidance_min':  guidance.min(),
            'q_data_mean':   q_data.mean(),
            'q_ref_mean':    q_ref.mean(),
            'q_scale':       q_scale,
        }
        return loss, info

    # ---- One-step policy --------------------------------------------------

    def one_step_actions(self, observations, noises, grad_params=None):
        return self.network.select('actor_onestep')(
            observations, noises, params=grad_params,
        )

    # ---- Entropic OT (log-domain Sinkhorn) --------------------------------

    def _cost_matrix(self, student_actions, teacher_actions):
        # Squared-L2 cost, normalized by batchwise mean (paper: divide by c_bar).
        diff = student_actions[:, :, None, :] - teacher_actions[:, None, :, :]
        cost = jnp.sum(diff * diff, axis=-1)
        scale = jnp.mean(cost, axis=(-2, -1), keepdims=True) + 1e-8
        return cost / scale

    def _sinkhorn(self, cost, p, w, eps_ot, iters, clip=1e-8):
        B, N, M = cost.shape
        p = jnp.clip(p, clip, 1.0)
        p = p / p.sum(-1, keepdims=True)
        w = jnp.clip(w, clip, 1.0)
        w = w / w.sum(-1, keepdims=True)

        logK = -cost / eps_ot
        log_p = jnp.log(p)
        log_w = jnp.log(w)

        log_u = jnp.zeros((B, N), dtype=cost.dtype)
        log_v = jnp.zeros((B, M), dtype=cost.dtype)

        def body(_, state):
            lu, lv = state
            logKv = jsp.special.logsumexp(logK + lv[:, None, :], axis=-1)
            lu = log_p - logKv
            logKTu = jsp.special.logsumexp(logK + lu[:, :, None], axis=-2)
            lv = log_w - logKTu
            return lu, lv

        log_u, log_v = jax.lax.fori_loop(0, iters, body, (log_u, log_v))

        logP = logK + log_u[:, :, None] + log_v[:, None, :]
        return jnp.exp(logP)

    def _hard_anchor(self, P, teacher_actions):
        # Row-wise hard assignment: j_i* = argmax_j P*_{ij}.
        j_star = jnp.argmax(P, axis=-1)
        selected_mass = jnp.take_along_axis(
            P, j_star[..., None], axis=-1,
        )[..., 0]
        anchor = jax.vmap(lambda a, idx: a[idx])(teacher_actions, j_star)
        return anchor, selected_mass, j_star

    def _teacher_marginal(self, q_teacher):
        # Value-weighted reference marginal q_j = softmax(Q / tau).
        tau = jnp.asarray(self.config['w_temperature'], dtype=q_teacher.dtype)
        return jax.nn.softmax(q_teacher / tau, axis=-1)

    # ---- Actor loss -------------------------------------------------------

    def actor_loss(self, batch, grad_params, rng):
        B, act_dim = batch['actions'].shape
        obs = batch['observations']
        ob_shape = obs.shape[1:]
        N = self.config['n_student']
        M = self.config['n_teacher']

        rng, vabc_rng, z_rng = jax.random.split(rng, 3)

        vabc_flow_loss, vabc_info = self.vabc_loss(batch, grad_params, vabc_rng)

        sg_params = jax.tree_util.tree_map(jax.lax.stop_gradient, grad_params)

        rng, zs_rng, zt_rng = jax.random.split(z_rng, 3)
        z_s = jax.random.normal(zs_rng, (B, N, act_dim))
        z_t = jax.random.normal(zt_rng, (B, M, act_dim))

        obs_rep_s = jnp.broadcast_to(
            obs[:, None], (B, N) + ob_shape,
        ).reshape(B * N, *ob_shape)
        z_flat_s = z_s.reshape(B * N, act_dim)
        s_flat = self.one_step_actions(obs_rep_s, z_flat_s, grad_params=grad_params)
        student_actions = jnp.clip(s_flat, -1, 1).reshape(B, N, act_dim)

        obs_rep_t = jnp.broadcast_to(
            obs[:, None], (B, M) + ob_shape,
        ).reshape(B * M, *ob_shape)
        z_flat_t = z_t.reshape(B * M, act_dim)
        t_flat = self._sample_from_flow(obs_rep_t, z_flat_t, params=grad_params)
        teacher_actions = jax.lax.stop_gradient(
            jnp.clip(t_flat, -1, 1).reshape(B, M, act_dim)
        )

        q_teacher = self._critic_agg(
            obs_rep_t, teacher_actions.reshape(B * M, act_dim), sg_params,
        ).reshape(B, M)
        q_student = self._critic_agg(
            obs_rep_s, student_actions.reshape(B * N, act_dim), sg_params,
        ).reshape(B, N)

        p = jnp.ones((B, N), dtype=q_teacher.dtype) / N
        q = jax.lax.stop_gradient(self._teacher_marginal(q_teacher))

        cost = self._cost_matrix(student_actions, teacher_actions)
        P = self._sinkhorn(
            cost, p, q,
            eps_ot=self.config['sinkhorn_eps'],
            iters=self.config['sinkhorn_iters'],
        )
        P = jax.lax.stop_gradient(P)

        anchor, selected_mass, selected_j = self._hard_anchor(P, teacher_actions)
        anchor = jax.lax.stop_gradient(anchor)

        sq = jnp.sum((student_actions - anchor) ** 2, axis=-1)
        distill_loss = jnp.sum(selected_mass * sq, axis=1).mean()

        qmax_loss = -jnp.sum(selected_mass * q_student, axis=1).mean()
        lambda_q = self.config.get('lambda_q', 0.0)

        actor_loss = (
            self.config['lambda_vbc']     * vabc_flow_loss
            + self.config['lambda_distill'] * distill_loss
            + lambda_q                      * qmax_loss
        )

        q_selected = jnp.take_along_axis(
            jnp.broadcast_to(q_teacher[:, None, :], (B, N, M)),
            selected_j[..., None], axis=-1,
        )[..., 0]

        metrics = {
            'loss':            actor_loss,
            'vabc/loss':       vabc_flow_loss,
            'distill/loss':    distill_loss,
            'vabc/guidance_mean': vabc_info['guidance_mean'],
            'vabc/guidance_max':  vabc_info['guidance_max'],
            'vabc/guidance_min':  vabc_info['guidance_min'],
            'vabc/q_data_mean':   vabc_info['q_data_mean'],
            'vabc/q_ref_mean':    vabc_info['q_ref_mean'],
            'vabc/q_scale':       vabc_info['q_scale'],
            'q/selected_mean':    q_selected.mean(),
            'q/student_mean':     q_student.mean(),
            'q/teacher_mean':     q_teacher.mean(),
            'q/teacher_minus_student': (q_teacher.mean() - q_student.mean()),
            'qmax/loss':          qmax_loss,
            'qmax/lambda_q':      jnp.asarray(lambda_q, dtype=q_teacher.dtype),
            'transport/selected_mass_mean': selected_mass.mean(),
        }
        return actor_loss, metrics

    # ---- Combined loss & update -------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, act_rng, crit_rng = jax.random.split(rng, 3)
        c_loss, c_info = self.critic_loss(batch, grad_params, crit_rng)
        a_loss, a_info = self.actor_loss(batch, grad_params, act_rng)
        info = {}
        for k, v in c_info.items(): info[f'critic/{k}'] = v
        for k, v in a_info.items(): info[f'actor/{k}'] = v
        return c_loss + a_loss, info

    def target_update(self, network, module_name):
        tau = self.config['tau']
        new_t = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        p = flax.core.unfreeze(network.params)
        p[f'modules_target_{module_name}'] = new_t
        return network.replace(params=flax.core.freeze(p))

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_actor_only(self, batch):
        # Actor-only step (paper Tab. 2 'every 5 steps' override for antsoccer):
        # actor gradient flows; critic gradient is zero by construction (actor
        # loss reads critic params under stop_gradient), so critic params don't
        # move and target-critic Polyak is skipped.
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            a_loss, a_info = self.actor_loss(batch, grad_params, rng)
            info = {f'actor/{k}': v for k, v in a_info.items()}
            return a_loss, info
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ---- Inference --------------------------------------------------------

    @jax.jit
    def sample_actions(self, observations, seed=None, grad_params=None):
        params = self.network.params if grad_params is None else grad_params
        noises = jax.random.normal(
            seed,
            (*observations.shape[:-len(self.config['ob_dims'])], self.config['action_dim']),
        )
        return self._sample_from_onestep(observations, noises, params=params)

    @jax.jit
    def sample_teacher_actions(self, observations, seed=None, grad_params=None):
        params = self.network.params if grad_params is None else grad_params
        noises = jax.random.normal(
            seed,
            (*observations.shape[:-len(self.config['ob_dims'])], self.config['action_dim']),
        )
        return self._sample_from_flow(observations, noises, params=params)

    # ---- Construction -----------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_times   = ex_actions[..., :1]
        ob_dims    = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            enc = encoder_modules[config['encoder']]
            encoders['critic']        = enc()
            encoders['actor_bc']      = enc()
            encoders['actor_onestep'] = enc()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_def = FlowPolicy(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc'),
        )
        actor_onestep_def = NNPolicy(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['one_step_tanh_squash'],
            final_fc_init_scale=config['one_step_fc_scale'],
            encoder=encoders.get('actor_onestep'),
        )

        net_info = dict(
            critic        = (critic_def,                (ex_observations, ex_actions)),
            target_critic = (copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc      = (actor_bc_def,              (ex_observations, ex_actions, ex_times)),
            actor_onestep = (actor_onestep_def,         (ex_observations, ex_actions)),
        )

        net_def  = ModuleDict({k: v[0] for k, v in net_info.items()})
        net_args = {k: v[1] for k, v in net_info.items()}

        if config['max_grad_norm'] is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optax.adam(learning_rate=config['lr']),
            )
        else:
            tx = optax.adam(learning_rate=config['lr'])

        params  = flax.core.freeze(net_def.init(init_rng, **net_args)['params'])
        network = TrainState.create(net_def, params, tx=tx)

        p = flax.core.unfreeze(network.params)
        p['modules_target_critic'] = p['modules_critic']
        network = network.replace(params=flax.core.freeze(p))

        config['ob_dims']    = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
