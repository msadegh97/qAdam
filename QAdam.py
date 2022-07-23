import torch
from torch.optim import _functional as F

from torch.optim.optimizer import Optimizer

import pdb
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

class QAdam(Optimizer):
    r"""Implements Adam algorithm with quantized states.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: amsgrad                      \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 representation='fp',
                 rounding='nearest',
                 bitwidth=8,
                 scheme='absmax',
                 **bitwidth_kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

        Optimizer.__init__(self, params, defaults)

        self.bitwidth_kwargs = bitwidth_kwargs
        self.rounding = rounding
        self.scheme = scheme
        self.representation= representation
    def __setstate__(self, state):
        Optimizer.__setstate__(self, state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def quant_normalization_scale(self, input_tensor):
        if self.scheme == 'absmax':

            mxAbs = torch.max(torch.abs(input_tensor))
            min_val = torch.min(input_tensor)

            if self.representation == 'rdx4':
                gradExp = torch.ceil(torch.log2(mxAbs) / 2)
                mxExp = 3
                curr_scale = 4 ** (gradExp - mxExp)
            elif self.representation == 'rdx2':
                gradExp = torch.ceil(torch.log2(mxAbs))
                mxExp = 3
                curr_scale = 2 ** (gradExp - mxExp)
            elif self.representation == 'fp':
                gradExp = torch.ceil(torch.log2(mxAbs))
                mxExp = 2 ** (self.bitwidth_kwargs['exp'] - 1) - 1
                curr_scale = 2 ** (gradExp - mxExp)
            elif self.representation == 'int':
                if min_val.item() < 0:
                    curr_scale = mxAbs / (2 ** (self.bitwidth - 1) - 1)
                else:
                    curr_scale = mxAbs / (2 ** (self.bitwidth) - 1 )

            scale = max(
                curr_scale,
                3.8147e-06)  # to make sure that we do not have overflow!

        return scale

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])


            F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])

            for p, exp_avgs_buffer, exp_avg_sqs_buffer in zip(params_with_grad, exp_avgs, exp_avg_sqs):
                state = self.state[p]
                
                if self.bitwidth_kwargs['scaling'] == 'simple':
                    m = self.bitwidth_kwargs['man']
                    e = self.bitwidth_kwargs['exp']
                    max_representable_fm = (2 - (0.5) ** (m)) * 2 ** (2 ** (e - 1) - 1)
                    max_representable_sm = (2 - (0.5) ** (m)) * 2 ** (2 ** (e) - 1)
                    
                    exp_avgs_buffer_scale =  torch.max(exp_avgs_buffer) / max_representable_fm
                    exp_avg_sqs_buffer_scale = torch.max(exp_avg_sqs_buffer) / max_representable_sm
                else:    
                    exp_avgs_buffer_scale = self.quant_normalization_scale(exp_avgs_buffer)
                    exp_avg_sqs_buffer_scale = self.quant_normalization_scale(exp_avg_sqs_buffer)
                # pdb.set_trace()

                if not self.bitwidth_kwargs['onlyqsm']:
                    state['exp_avg'] = float_quantize(exp_avgs_buffer / exp_avgs_buffer_scale,
                                                        exp = self.bitwidth_kwargs['exp'],
                                                        man= self.bitwidth_kwargs['man'],
                                                        rounding = self.rounding) * exp_avgs_buffer_scale
                if not self.bitwidth_kwargs['onlyqfm']:
                    state['exp_avg_sq'] = float_quantize(exp_avg_sqs_buffer / exp_avg_sqs_buffer_scale,
                                                            exp = self.bitwidth_kwargs['exp'] + 1,
                                                            man= self.bitwidth_kwargs['man'], 
                                                            rounding= self.rounding) * exp_avg_sqs_buffer_scale 
        return loss
