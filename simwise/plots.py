import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_vector_plotly(t_arr, data, titles, ylabel="Value"):
    """Helper function to plot vector quantities"""
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=titles,
                       shared_xaxes=True,
                       vertical_spacing=0.1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Plotly default colors
    
    for i in range(3):
        fig.add_trace(
            go.Scatter(x=t_arr, y=data[:, i], line=dict(color=colors[i])),
            row=i+1, col=1
        )
        fig.update_yaxes(title_text=ylabel, row=i+1, col=1)
    
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_layout(showlegend=False, height=800)
    
    return fig

def plot_quaternion_plotly(t_arr, quaternions):
    """Helper function to plot quaternions"""
    fig = make_subplots(rows=4, cols=1, 
                       subplot_titles=['w', 'x', 'y', 'z'],
                       shared_xaxes=True,
                       vertical_spacing=0.05)
    
    for i in range(4):
        fig.add_trace(
            go.Scatter(x=t_arr, y=quaternions[:, i], 
                      line=dict(color='#1f77b4')),
            row=i+1, col=1
        )
        fig.update_yaxes(title_text=f"q{i}", row=i+1, col=1)
    
    fig.update_xaxes(title_text="Time [s]", row=4, col=1)
    fig.update_layout(showlegend=False, height=1000)
    
    return fig
