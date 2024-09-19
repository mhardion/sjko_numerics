import torch
import plotly.graph_objects as go
from .utils import xy, xyz
from .functionals import *

def go_mass_flow(mass_flow, F, **kwargs):
    dom = F.domain.to('cpu')
    zmax = mass_flow.max().item()
    return [go.Heatmap(x=dom[0,:,0], y=dom[:,0,1], z=µ, zmin=0, zmax=zmax, **kwargs) for µ in mass_flow]

def go_traj(traj, **kwargs):
    color = kwargs.pop('color', 'blue')
    line = kwargs.pop('line', {})
    line['color'] = color
    marker = kwargs.pop('marker', {})
    size = marker.get('size', 10)
    marker['color'] = color
    flow_lines = [go.Scatter3d(xyz(traj[:i+1]),
                               mode='lines+markers',
                               marker=dict(marker, size=i*[0]+[size]),
                               line=line,
                               **kwargs) for i in range(traj.size(0))]
    return flow_lines

def go_sphere(N=100, heatmap=None, **kwargs):
    theta = torch.linspace(0, 2*torch.pi, N)
    phi = torch.linspace(0, torch.pi, N//2)
    x = torch.outer(torch.cos(theta), torch.sin(phi))
    y = torch.outer(torch.sin(theta), torch.sin(phi))
    z = torch.outer(torch.ones(theta.size()), torch.cos(phi))
    if heatmap is not None:
        xyz = torch.cat((x[...,None], y[...,None], z[...,None]), dim=-1)
        h = heatmap(xyz)
        return go.Surface(x=x, y=y, z=z, surfacecolor=h, **kwargs)
    return go.Surface(x=x, y=y, z=z, **kwargs)

def go_figwithbuttons3d(data, fig_side_px=700, dt=100, axisrange=[-1, 1], axisvisible=True, title=""):
    return go.Figure(data=data,
                     layout=go.Layout(
                         height=fig_side_px,
                         width=fig_side_px,
                         scene=dict(
                         xaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         yaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         zaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         aspectmode='cube'),
                         title=title,
                         updatemenus=[dict(type="buttons",
                                           showactive=False,
                                           buttons=[dict(label='Play',
                                                         method='animate',
                                                         args=[None, dict(frame=dict(duration=dt, redraw=True),
                                                                            fromcurrent=True)]),
                                                    dict(label='Pause',
                                                         method='animate',
                                                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                            mode='immediate',
                                                                            transition=dict(duration=0))])])]
                    ),
            )

def go_figwithbuttons2d(data, fig_side_px=700, dt=100, axisrange=[-1, 1], axisvisible=True, title=""):
    return go.Figure(data=data,
                     layout=go.Layout(
                         height=fig_side_px,
                         width=fig_side_px,
                         xaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         yaxis=dict(range=axisrange, autorange=False, visible=axisvisible),
                         title=title,
                         updatemenus=[dict(type="buttons",
                                           showactive=False,
                                           buttons=[dict(label='Play',
                                                         method='animate',
                                                         args=[None, dict(frame=dict(duration=dt, redraw=True),
                                                                            fromcurrent=True)]),
                                                    dict(label='Pause',
                                                         method='animate',
                                                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                            mode='immediate',
                                                                            transition=dict(duration=0))])])]
                    ),
            )

def potential_heatmap(V, domain, grid_size=50, **kwargs):
    xm, xM, ym, yM = domain
    x = torch.linspace(xm, xM, grid_size)
    y = torch.linspace(ym, yM, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xxx = torch.cat((xx[...,None], yy[...,None]), dim=-1)
    return go.Heatmap(x=x, y=y, z=V(xxx.reshape(-1, 2)).reshape(grid_size, grid_size), **kwargs)