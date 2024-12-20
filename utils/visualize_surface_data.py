from flask import Flask, render_template_string
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import json
import argparse

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Point Cloud Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { margin: 0; }
            #plot { width: 100vw; height: 100vh; }
        </style>
    </head>
    <body>
        <div id="plot"></div>
        <script>
            var graphs = {{graphJSON | safe}};
            Plotly.newPlot('plot', graphs.data, graphs.layout);
        </script>
    </body>
</html>
'''

def create_arrow_traces(points, normals, sample_size=500, scale_factor=0.03):  # Reduced scale_factor from 0.1 to 0.05
    """Create arrow traces for sampled normal vectors"""
    # Randomly sample indices
    indices = np.random.choice(len(points), sample_size, replace=False)
    sampled_points = points[indices]
    sampled_normals = normals[indices]
    
    # Create arrow heads (cones)
    cone_traces = []
    
    # Create lines for arrow shafts
    shaft_x = []
    shaft_y = []
    shaft_z = []
    
    for i in range(len(sampled_points)):
        # Start and end points of arrow shaft
        start = sampled_points[i]
        end = start + sampled_normals[i] * scale_factor
        
        # Add shaft lines
        shaft_x.extend([start[0], end[0], None])
        shaft_y.extend([start[1], end[1], None])
        shaft_z.extend([start[2], end[2], None])
        
        # Create small cone at the end for arrow head
        direction = sampled_normals[i]
        
        cone = go.Cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[direction[0]],
            v=[direction[1]],
            w=[direction[2]],
            sizeref=0.05,  # Reduced sizeref from 0.1 to 0.05
            sizemode='absolute',
            showscale=False,
            colorscale=[[0, 'red'], [1, 'red']],
            anchor="tip"
        )
        cone_traces.append(cone)
    
    # Create shaft trace
    shaft_trace = go.Scatter3d(
        x=shaft_x,
        y=shaft_y,
        z=shaft_z,
        mode='lines',
        line=dict(color='red', width=2),
        name='Normal Vectors'
    )
    
    return [shaft_trace] + cone_traces

@app.route('/')
def index():
    # Load the NPZ file
    data = np.load(app.config['FILE_PATH'])
    surface = data['surface']

    # Split into points and normals
    points = surface[:, :3]
    normals = surface[:, 3:]

    # Create the main scatter plot for points
    points_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',  # Changed from lightblue to blue
            opacity=0.8
        ),
        name='Points'
    )

    # Create arrow traces for sampled normals
    normal_traces = create_arrow_traces(points, normals, sample_size=500, scale_factor=0.05)  # Reduced scale_factor from 0.1 to 0.05

    # Combine all traces
    traces = [points_trace] + normal_traces

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title='Point Cloud with Normal Vectors (500 samples)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True
    )

    # Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template_string(HTML_TEMPLATE, graphJSON=graphJSON)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize point cloud from NPZ file')
    parser.add_argument('--file_path', type=str, required=True, help='Path to NPZ file containing surface data')
    args = parser.parse_args()
    
    app.config['FILE_PATH'] = args.file_path
    app.run(host='0.0.0.0', port=8081, debug=True)