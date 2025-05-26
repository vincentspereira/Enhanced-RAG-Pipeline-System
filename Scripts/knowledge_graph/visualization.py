"""
Knowledge graph visualization module.
"""
from typing import List, Dict, Any, Tuple, Set, Optional
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import tempfile
from plotly import graph_objects as go
import pandas as pd
from pyvis.network import Network

class KnowledgeGraphVisualizer:
    """Visualizer for knowledge graphs with multiple rendering options."""
    
    def __init__(self, output_dir: str = "data/visualizations"):
        """Initialize the knowledge graph visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_matplotlib(self, graph: nx.DiGraph, filename: str, 
                            title: str = "Knowledge Graph",
                            figsize: Tuple[int, int] = (12, 10)):
        """Visualize the graph using matplotlib.
        
        Args:
            graph: NetworkX graph to visualize
            filename: Output filename (without extension)
            title: Title for the visualization
            figsize: Figure size as (width, height)
        
        Returns:
            Path to the saved visualization
        """
        plt.figure(figsize=figsize)
        plt.title(title)
        
        # Node positions
        pos = nx.spring_layout(graph, seed=42)
        
        # Node colors based on type
        node_types = set([graph.nodes[node].get('type', 'Unknown') for node in graph.nodes()])
        type_to_color = {t: plt.cm.tab10(i) for i, t in enumerate(node_types)}
        node_colors = [type_to_color[graph.nodes[node].get('type', 'Unknown')] for node in graph.nodes()]
        
        # Labels for nodes
        labels = {node: f"{node}\n({graph.nodes[node].get('type', 'Unknown')})" for node in graph.nodes()}
        
        # Draw the graph
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
        
        # Get unique edge types for the legend
        edge_types = set()
        for u, v, data in graph.edges(data=True):
            edge_types.add(data.get('predicate', 'unknown'))
        
        # Draw edges with different styles per type
        for i, edge_type in enumerate(edge_types):
            edges_to_draw = [(u, v) for u, v, data in graph.edges(data=True) 
                             if data.get('predicate', 'unknown') == edge_type]
            
            nx.draw_networkx_edges(
                graph, pos, 
                edgelist=edges_to_draw,
                width=1.0, 
                alpha=0.7,
                edge_color=plt.cm.tab10(i),
                style='solid' if i % 2 == 0 else 'dashed',
                arrows=True,
                arrowstyle='-|>', 
                arrowsize=10
            )
        
        # Legend for node types
        node_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, 
                                  label=f'Node: {node_type}')
                       for node_type, color in type_to_color.items()]
        
        # Legend for edge types
        edge_patches = [plt.Line2D([0], [0], color=plt.cm.tab10(i), 
                                  lw=2, linestyle='solid' if i % 2 == 0 else 'dashed',
                                  label=f'Edge: {edge_type}')
                       for i, edge_type in enumerate(edge_types)]
        
        plt.legend(handles=node_patches + edge_patches, loc='upper right', 
                  bbox_to_anchor=(1.3, 1.0))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_plotly(self, graph: nx.DiGraph, filename: str,
                         title: str = "Interactive Knowledge Graph"):
        """Create an interactive visualization using Plotly.
        
        Args:
            graph: NetworkX graph to visualize
            filename: Output filename (without extension)
            title: Title for the visualization
        
        Returns:
            Path to the saved visualization
        """
        # Create a Plotly figure
        fig = go.Figure()
        
        # Node positions
        pos = nx.spring_layout(graph, seed=42)
        
        # Node types with different colors
        node_types = set([graph.nodes[node].get('type', 'Unknown') for node in graph.nodes()])
        type_to_color = {t: f'rgb{tuple(int(c*255) for c in plt.cm.tab10(i)[:3])}' 
                         for i, t in enumerate(node_types)}
        
        # Edge data
        edge_x = []
        edge_y = []
        edge_trace = []
        
        # Group edges by predicate
        edges_by_predicate = {}
        for edge in graph.edges(data=True):
            predicate = edge[2].get('predicate', 'unknown')
            if predicate not in edges_by_predicate:
                edges_by_predicate[predicate] = []
            edges_by_predicate[predicate].append(edge)
        
        # Create a trace for each predicate type
        for i, (predicate, edges) in enumerate(edges_by_predicate.items()):
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Add a slight curve to distinguish bidirectional connections
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)  # Break the line
                
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)  # Break the line
            
            color = f'rgb{tuple(int(c*255) for c in plt.cm.tab10(i)[:3])}'
            edge_trace.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color=color),
                hoverinfo='none',
                mode='lines',
                name=f'Relation: {predicate}'
            ))
        
        # Add edge traces to figure
        for trace in edge_trace:
            fig.add_trace(trace)
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text with node properties
            properties = graph.nodes[node]
            hover_text = f"<b>ID:</b> {node}<br>"
            hover_text += f"<b>Type:</b> {properties.get('type', 'Unknown')}<br>"
            
            for key, value in properties.items():
                if key != 'type':
                    hover_text += f"<b>{key}:</b> {value}<br>"
            
            node_text.append(hover_text)
            node_color.append(type_to_color[properties.get('type', 'Unknown')])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_color,
                size=10,
                line=dict(width=1, color='rgb(50,50,50)')
            ),
            name='Entities'
        )
        
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=title,
            titlefont=dict(size=16),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                itemsizing='constant'
            )
        )
        
        # Save to HTML
        output_path = os.path.join(self.output_dir, f"{filename}.html")
        fig.write_html(output_path)
        
        return output_path
    
    def visualize_pyvis(self, graph: nx.DiGraph, filename: str,
                        title: str = "Interactive Knowledge Graph",
                        height: str = "800px", width: str = "100%"):
        """Create an interactive visualization using PyVis.
        
        Args:
            graph: NetworkX graph to visualize
            filename: Output filename (without extension)
            title: Title for the visualization
            height: Height of the visualization
            width: Width of the visualization
        
        Returns:
            Path to the saved visualization
        """
        # Initialize PyVis network
        net = Network(height=height, width=width, directed=True, notebook=False)
        net.heading = title
        
        # Get node types and assign colors
        node_types = set([graph.nodes[node].get('type', 'Unknown') for node in graph.nodes()])
        type_to_color = {t: f'#{int(plt.cm.tab10(i)[0]*255):02x}{int(plt.cm.tab10(i)[1]*255):02x}{int(plt.cm.tab10(i)[2]*255):02x}' 
                         for i, t in enumerate(node_types)}
        
        # Add nodes to the network
        for node_id in graph.nodes():
            properties = graph.nodes[node_id]
            node_type = properties.get('type', 'Unknown')
            
            # Create title (hover text)
            title_text = f"ID: {node_id}<br>Type: {node_type}<br>"
            for key, value in properties.items():
                if key != 'type':
                    title_text += f"{key}: {value}<br>"
            
            net.add_node(
                node_id, 
                label=str(node_id), 
                title=title_text,
                color=type_to_color[node_type],
                shape='dot' if node_type in ['Person', 'Organization'] else 'square'
            )
        
        # Add edges to the network
        for source, target, data in graph.edges(data=True):
            predicate = data.get('predicate', 'unknown')
            
            # Create edge title
            title_text = f"Relation: {predicate}<br>"
            for key, value in data.items():
                if key != 'predicate':
                    title_text += f"{key}: {value}<br>"
            
            net.add_edge(
                source, 
                target, 
                label=predicate,
                title=title_text,
                arrows='to'
            )
        
        # Add legend for node types
        legend_html = """
        <div style="padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9; margin-top: 10px;">
            <h3>Legend</h3>
            <h4>Node Types</h4>
            <ul style="list-style-type: none; padding-left: 0;">
        """
        
        for node_type, color in type_to_color.items():
            legend_html += f'<li><span style="display: inline-block; width: 15px; height: 15px; background-color: {color};"></span> {node_type}</li>'
        
        legend_html += """
            </ul>
        </div>
        """
        
        # Configure physics
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.001,
            damping=0.09
        )
        
        # Save the network
        output_path = os.path.join(self.output_dir, f"{filename}.html")
        net.save_graph(output_path)
        
        # Add legend to the HTML
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Insert legend before the closing body tag
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Generate statistics for the knowledge graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of statistics
        """
        # Basic graph statistics
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": nx.is_directed(graph),
        }
        
        # Node type distribution
        node_types = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats["node_type_distribution"] = node_types
        
        # Edge type distribution
        edge_types = {}
        for _, _, data in graph.edges(data=True):
            edge_type = data.get('predicate', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats["edge_type_distribution"] = edge_types
        
        # Centrality measures for top nodes
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            stats["top_degree_centrality"] = [{"node": node, "value": round(value, 4)} for node, value in top_degree]
            
            # Betweenness centrality (only if graph is connected enough)
            if stats["density"] > 0.05:
                betweenness = nx.betweenness_centrality(graph)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                stats["top_betweenness_centrality"] = [{"node": node, "value": round(value, 4)} for node, value in top_betweenness]
        except:
            stats["centrality_error"] = "Unable to calculate centrality measures, graph may be too sparse"
        
        # Connected components
        if not nx.is_directed(graph):
            stats["num_connected_components"] = nx.number_connected_components(graph)
        else:
            stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(graph)
            stats["num_strongly_connected_components"] = nx.number_strongly_connected_components(graph)
        
        return stats
"""
