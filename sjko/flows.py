# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.colors import LogNorm, Normalize
import plotly.graph_objects as go
from .utils import *
from .functionals import *
from .visualization import *
from .sinkhorn import *

class FlowResult():
    def __init__(self):
        self.time = torch.zeros(0)
        self.stats = {}

    def plot_stats(self, stat, ax, **kwargs):
        dl = len(self.time)-len(self.stats[stat])
        if  dl >= 0:
            ax.plot(self.time[dl:], self.stats[stat], **kwargs)
        else:
            ax.plot(self.stats[stat], **kwargs)
        return ax

    def _fig_ax_config(self, fig_side_inches=5, projection=None):
        # Configure fig and axes
        fig = plt.figure()
        fig.set_size_inches((fig_side_inches, fig_side_inches/.9))
        ax = fig.add_subplot(projection=projection, position=[0., .1, 1., .9])
        # time display
        time_ax = plt.Axes(fig, [0., 0., 1, 0.1])
        time_ax.set_axis_off()
        time_ax.set(xlim=(-.2,1.2), ylim=(-.1, .1))
        time_ax.plot([0, 1], [0, 0],'k')
        time_ax.text(-.01, 0., "0.0", ha='right', va="center")
        time_ax.text(1.01, 0., f"{self.time[-1]:.2E}", va="center")
        fig.add_axes(ax)
        fig.add_axes(time_ax)
        return fig, ax, time_ax

    def _append_stats(self, tau, stats):
        if not len(self.time):
            self.time = torch.zeros(1)
        else:
            self.time = torch.cat((self.time, self.time[-1][None]+tau))
        for key in stats.keys():
            if key not in self.stats.keys():
                self.stats[key] = [stats[key]]
            else:
                self.stats[key].append(stats[key])

class EulerianFlowResult(FlowResult):

    def __init__(self, F):
        super().__init__()
        self.F = F
        self.mass_flow = None

    def append(self, tau, a, stats={}):
        self._append_stats(tau, stats)
        a_ = a.reshape(self.F.domain.size()[:-1]).to('cpu')
        self.mass_flow = (a_[None,...] if self.mass_flow is None 
                     else torch.cat([self.mass_flow, a_[None,...]], dim=0))
    
    def go_anim_flow(self, subsample_ratio=1):
        data = go_mass_flow(self.mass_flow[::subsample_ratio], self.F)
        fig = go_figwithbuttons([data[0]])
        fig.frames = [go.Frame(data=[d]) for d in data]
        return fig
    
    def go_flowdata_simplex(self, subsample_ratio=1, **kwargs):
        return go_traj(self.mass_flow[::subsample_ratio], **kwargs)
    
    def go_anim_simplex(self, duration, subsample_ratio=1, fig_side_px=700, n=100, **kwargs):
        dt = subsample_ratio*duration/self.mass_flow.size(0)
        simplex = self.go_simplexdata(n)
        flow = self.go_flowdata_simplex(subsample_ratio)
        fig = go_figwithbuttons(simplex+[flow[0]], fig_side_px, dt)
        fig.frames = [go.Frame(data=simplex + [flow[i]])
                          for i in range(len(flow))]
        return fig
    
class EulerianSinkhornFlowResult(EulerianFlowResult):
    def __init__(self, F, eps):
        super().__init__(F)
        self.eps = eps
        self.Hc = torch.exp(-F.c/eps)
        self.b_flow = None

    def append(self, tau, µ, f_µ, stats={}):
        super().append(tau, µ, stats)
        b = torch.exp(-f_µ/self.eps).to('cpu')
        self.b_flow = (b[None,...] if self.b_flow is None 
                else torch.cat([self.b_flow, b[None,...]], dim=0))
    
    def append_b(self, tau, b, stats={}):
        µ = b*torch.linalg.lstsq(self.Hc, b).solution
        super().append(tau, µ, stats)
        self.b_flow = (b.to('cpu')[None,...] if self.b_flow is None 
                else torch.cat([self.b_flow, b.to('cpu')[None,...]], dim=0))

    def go_spheredata(self, n=100, B_kwargs={}, sphere_kwargs={}):
        P, Q = apply_to_eig(self.Hc, lambda l:1/torch.sqrt(l), torch.sqrt)
        t = torch.linspace(0, 1, n)[:,None]
        t_ = 1-t
        B = torch.cat([t_*Q[:,i] + t*Q[:,(i+1)%3] for i in range(3)])
        B /= torch.sqrt((B*B).sum(-1)[:,None])
        Bkwargs = dict(dict(mode='lines', line=dict(width=1, color='red'), name=r'Boundary of B'), **B_kwargs)
        b_line = go.Scatter3d(xyz(B), **Bkwargs)
        data = [b_line]
        if isinstance(self.F, PotentialEnergy):
            V = torch.diag(self.F.potential)
            PVQ = P @ V @ Q
            heatmap = lambda b: torch.einsum('ijk,kk,ijk->ij', b, PVQ, b)
            data.append(go_sphere(heatmap=heatmap, **dict(showscale=False, **sphere_kwargs)))
            A = (2/self.eps)*(Q @ V @ P - P @ V @ Q)
            a = torch.tensor([A[2, 1], A[0, 2], A[1, 0]])
            a /= torch.sqrt(sqnorm(a))
            I = torch.eye(3)
            k = sqnorm(I-a).argmax()
            e = I[k,:]
            v = a - e
            R = I - (2/sqnorm(v))*torch.outer(v, v)
            mask = torch.ones(3).type(torch.bool)
            mask[k] = False
            basis = R[mask,:]
            basis /= torch.sqrt(sqnorm(basis))[:,None]
            ls = torch.linspace(-1, 1, 20)
            rs = torch.sqrt(1-ls*ls)
            theta = torch.linspace(0, 2*torch.pi, n)[:,None]
            c, s = torch.cos(theta), torch.sin(theta)
            for l, r in zip(ls, rs):
                circ = l*a + r*c*basis[0] + r*s*basis[1]
                data.append(go.Scatter3d(xyz(circ), mode='lines', line=dict(width=1, color='black'), showlegend=False))
        else:
            data.append(go_sphere(colorscale=[[0, 'gray'], [1, 'gray']], **dict(showscale=False, **sphere_kwargs)))
        return data

    def go_flowdata_b(self, subsample_ratio=1, **kwargs):
        P = apply_to_eig(self.Hc, lambda l:1/torch.sqrt(l))
        traj = self.b_flow[::subsample_ratio,:] @ P
        return go_traj(traj, **kwargs)



    def go_anim_b(self, subsample_ratio=1, dt=100, fig_side_px=700, axisvisible=True, n=100, B_kwargs={}, sphere_kwargs={}, traj_kwargs=dict(line=dict(width=5), marker=dict(size=10))):
        if self.Hc.size(0) == 3:
            sphere = self.go_spheredata(n, B_kwargs, sphere_kwargs)
            flow_lines = self.go_flowdata_b(subsample_ratio, **traj_kwargs)
            fig = go_figwithbuttons(sphere+[flow_lines[0]], fig_side_px, dt, axisvisible=axisvisible)
            fig.frames = [go.Frame(data=sphere + [flow_lines[i]])
                          for i in range(len(flow_lines))]
            return fig
        else:
            raise NotImplementedError





class LagrangianFlowResult(FlowResult):
        def __init__(self, potential=None):
            super().__init__()
            self.particle_flow = None
            self.potential = potential
        
        def append(self, tau, x, stats={}):
            self._append_stats(tau, stats)
            self.particle_flow = (x[None,...] if self.particle_flow is None 
                             else torch.cat([self.particle_flow, x[None,...]], dim=0))

        def particle_flowdata(self, subsample_ratio=1, **kwargs):
            traj = self.particle_flow[::subsample_ratio,...]
            return [go.Scatter(x=x[:,0], y=x[:,1], mode='markers', **kwargs) for x in traj]

        def anim(self, subsample_ratio, dt=100, fig_side_px=700, axisvisible=True, heatmap_grid_size=50, **kwargs):
            xm, xM, ym, yM = self.particle_flow[:,:,0].min().item(), self.particle_flow[:,:,0].max().item(), self.particle_flow[:,:,1].min().item(), self.particle_flow[:,:,1].max().item()
            x = torch.linspace(xm, xM, heatmap_grid_size)
            y = torch.linspace(ym, yM, heatmap_grid_size)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            xxx = torch.cat((xx[...,None], yy[...,None]), dim=-1)
            heatmap = go.Heatmap(x=x, y=y, z=self.potential(xxx.reshape(-1, 2)).reshape(heatmap_grid_size, heatmap_grid_size))
            particles = self.particle_flowdata(subsample_ratio, **kwargs)
            fig = go_figwithbuttons([heatmap, particles[0]], fig_side_px, dt, axisvisible=axisvisible)
            fig.frames = [go.Frame(data=[particles[i]]) for i in range(len(particles))]
            return fig

        def save_anim(self, filename, duration, subsample_ratio=1, fig_side_inches=5, levels=10, gridw=50):
            fig, ax, t_ax = self._fig_ax_config(fig_side_inches)
            ax.set_axis_off()
            ts = t_ax.scatter([0], [0], marker='o', color='k')
            T = self.time[-1]
            timestamps = self.time[::subsample_ratio]/T
            dt = (timestamps[1:] - timestamps[:-1]).mean()
            xm, xM, ym, yM = self.particle_flow[:,:,0].min().item(), self.particle_flow[:,:,0].max().item(), self.particle_flow[:,:,1].min().item(), self.particle_flow[:,:,1].max().item()
            if self.potential is not None:
                xx, yy = torch.meshgrid(torch.linspace(xm, xM, gridw), torch.linspace(ym, yM, gridw), indexing='xy')
                xxx = torch.cat((xx[...,None], yy[...,None]), dim=-1)
                ax.contourf(xx, yy, self.potential(xxx.reshape(-1, 2)).reshape(gridw, gridw), cmap='plasma', levels=levels)
            sp = ax.scatter(self.particle_flow[0,:,0], self.particle_flow[0,:,1], c='k')
            frames = len(self.particle_flow)
            def func(frame):
                sp.set_offsets(self.particle_flow[frame])
                ts.set_offsets([frame/len(self.time), 0])
                return sp
            anim = FuncAnimation(fig, func, frames=frames, interval=int(1e3*dt*duration))
            anim.save(filename)
            plt.close()

class LagrangianSinkhornFlowResult(LagrangianFlowResult):
    def __init__(self, eps, potential=None):
        super().__init__(potential)
        self.eps = eps
        self.b_flow = None
    
    def append(self, tau, x, f_aa, stats={}):
        super().append(tau, x, stats)
        b = torch.exp(-f_aa/self.eps)
        self.b_flow = (b[None,...] if self.b_flow is None 
                  else torch.cat([self.b_flow, b[None,...]], dim=0))

def SJKO_flow(µ0, F, time, eps, descent_tol=1e-3, descent_maxiter=20, lr=1e-2, sinkhorn_maxiter=20, sinkhorn_tol=1e-3, verbose=False):
    # Initialize
    µ = µ0.clone().flatten()
    µ_prev = µ.clone()
    res = EulerianSinkhornFlowResult(F, eps)
    c = F.c
    f_aa = self_transport(µ, c, eps, tol=sinkhorn_tol, maxiter=sinkhorn_maxiter).flatten()
    res.append(0, µ_prev.to('cpu'), f_aa, {'F': F(µ_prev).item()})
    tau = time[1:] - time[:-1]
    for i in range(len(time)-1):
        if verbose:
            print(f'\r JKO step {i}...', end='')
        k = 0
        iter_counter = torch.zeros(1)
        while (k==0 or ((grad - µ @ grad) < -descent_tol).any()) and k < descent_maxiter:
            f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
                µ,
                µ_prev,
                c,
                c,
                c,
                [eps]*sinkhorn_maxiter,
                init=(None if not i else [f_aa, g_bb, g_ab, f_ba]),
                tol=sinkhorn_tol,
                iter_counter=iter_counter
            )
            grad = (f_ba - f_aa).flatten() + 2*tau[i]*F.grad(µ)
            µ = simplex_proj(µ - lr*grad)
            k+=1
        µ_prev = µ.clone()
        res.append(tau[i], µ_prev.to('cpu'), f_aa.detach().flatten(),
                   {'F': F(µ_prev).item(),
                    'descent_iter': k,
                    'sinkhorn_iter': iter_counter.item()})
    return res

def explicit_lagrangian(x0, potential, time):
    x = x0.clone().reshape(-1, x0.size(-1))
    res = LagrangianFlowResult(potential)
    res.append(0, x0.clone(), stats={'F': potential(x.detach()).mean()})
    x.requires_grad = True
    for tau in time[1:]-time[:-1]:
        g = torch.autograd.grad(outputs=potential(x)[:,None], inputs=x, grad_outputs=torch.ones((len(x),1)))[0]
        x = x - tau*g
        res.append(tau, x.detach().reshape(x0.size()), stats={'F': potential(x.detach()).mean()})
    return res