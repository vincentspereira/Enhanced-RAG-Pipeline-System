"""
Command-line interface for managing prompts.
"""
import argparse
import sys
import os
import json
import yaml
import logging
import time
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Scripts.llm.prompt_management import (
    PromptTemplate, 
    get_prompt_library,
    get_prompt_optimizer,
    DefaultPromptTemplates
)

def list_templates(args):
    """List available prompt templates."""
    library = get_prompt_library()
    templates = library.list_templates()
    
    if not templates:
        print("No templates found in the library.")
        return
    
    print(f"\n{len(templates)} templates found:\n")
    
    for i, template in enumerate(templates, 1):
        print(f"{i}. {template['name']} (v{template['version']})")
        print(f"   Description: {template['description']}")
        print(f"   Variables: {', '.join(template['variables'])}")
        print()

def view_template(args):
    """View a prompt template."""
    library = get_prompt_library()
    template = library.get_template(args.name)
    
    if not template:
        print(f"Template '{args.name}' not found.")
        return
    
    print(f"\nTemplate: {template.name} (v{template.version})")
    print(f"Description: {template.description}")
    print(f"Variables: {', '.join(template.variables)}")
    print("\nTemplate content:")
    print("-" * 80)
    print(template.template)
    print("-" * 80)
    
    # If requested, show optimization info
    if args.show_performance:
        optimizer = get_prompt_optimizer()
        analysis = optimizer.analyze_template_performance(template.name)
        
        print("\nPerformance Analysis:")
        
        if "results_count" in analysis and analysis["results_count"] > 0:
            print(f"Based on {analysis['results_count']} tracked usages")
            
            if "metrics_stats" in analysis:
                print("\nMetrics:")
                for metric, stats in analysis["metrics_stats"].items():
                    print(f"  {metric}: {stats['mean']:.4f} (± {stats['std']:.4f}) [min: {stats['min']:.4f}, max: {stats['max']:.4f}]")
            
            if "trends" in analysis and analysis["trends"]:
                print("\nTrends:")
                for metric, trend in analysis["trends"].items():
                    direction = "↑" if trend["direction"] == "positive" else "↓"
                    print(f"  {metric}: {direction} ({trend['change']:.4f})")
        else:
            print("No performance data available for this template.")

def create_template(args):
    """Create a new prompt template."""
    library = get_prompt_library()
    
    # Check if template already exists
    if library.get_template(args.name):
        print(f"Template '{args.name}' already exists. Use update command to modify it.")
        return
    
    # Get template content
    template_content = ""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            print(f"Error reading template file: {e}")
            return
    else:
        print("Enter template content (Ctrl+D or Ctrl+Z on a new line to finish):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            template_content = "\n".join(lines)
    
    # Create template
    try:
        # Auto-detect variables if not specified
        variables = args.variables.split(",") if args.variables else None
        
        template = PromptTemplate(
            template=template_content,
            name=args.name,
            description=args.description,
            variables=variables,
            version="1.0"
        )
        
        # Add to library
        success = library.add_template(template)
        
        if success:
            print(f"Template '{args.name}' created successfully.")
            print(f"Detected variables: {', '.join(template.variables)}")
        else:
            print(f"Failed to create template '{args.name}'.")
    except Exception as e:
        print(f"Error creating template: {e}")

def update_template(args):
    """Update an existing prompt template."""
    library = get_prompt_library()
    
    # Check if template exists
    template = library.get_template(args.name)
    if not template:
        print(f"Template '{args.name}' not found.")
        return
    
    # Get template content
    template_content = template.template
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            print(f"Error reading template file: {e}")
            return
    elif args.edit:
        # Create temporary file with current content
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix=".txt", mode='w+', delete=False) as temp:
            temp.write(template_content)
            temp_path = temp.name
        
        # Open in editor
        editor = os.environ.get('EDITOR', 'notepad')  # Use notepad as default on Windows
        try:
            subprocess.call([editor, temp_path])
            
            # Read updated content
            with open(temp_path, 'r') as f:
                template_content = f.read()
                
            # Remove temp file
            os.unlink(temp_path)
        except Exception as e:
            print(f"Error editing template: {e}")
            return
    
    # Update description if provided
    description = args.description if args.description else template.description
    
    # Update variables if provided
    variables = args.variables.split(",") if args.variables else template.variables
    
    # Create updated template
    try:
        updated_template = PromptTemplate(
            template=template_content,
            name=args.name,
            description=description,
            variables=variables,
            version=template.version  # Will be auto-incremented
        )
        
        # Update in library
        success = library.update_template(args.name, updated_template)
        
        if success:
            print(f"Template '{args.name}' updated successfully to version {updated_template.version}.")
            print(f"Variables: {', '.join(updated_template.variables)}")
        else:
            print(f"Failed to update template '{args.name}'.")
    except Exception as e:
        print(f"Error updating template: {e}")

def delete_template(args):
    """Delete a prompt template."""
    library = get_prompt_library()
    
    # Check if template exists
    if not library.get_template(args.name):
        print(f"Template '{args.name}' not found.")
        return
    
    # Confirm deletion
    if not args.force:
        confirm = input(f"Are you sure you want to delete template '{args.name}'? (y/N): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return
    
    # Delete template
    success = library.delete_template(args.name)
    
    if success:
        print(f"Template '{args.name}' deleted successfully.")
    else:
        print(f"Failed to delete template '{args.name}'.")

def test_template(args):
    """Test a prompt template with provided variables."""
    library = get_prompt_library()
    
    # Check if template exists
    template = library.get_template(args.name)
    if not template:
        print(f"Template '{args.name}' not found.")
        return
    
    # Parse variables
    variables = {}
    if args.vars:
        for var_str in args.vars:
            try:
                name, value = var_str.split('=', 1)
                variables[name.strip()] = value.strip()
            except ValueError:
                print(f"Invalid variable format: {var_str}. Use name=value format.")
                return
    
    # Check for missing variables
    missing_vars = [var for var in template.variables if var not in variables]
    if missing_vars:
        print(f"Missing variables: {', '.join(missing_vars)}")
        
        # Prompt for missing variables
        for var in missing_vars:
            if var in ['context', 'document', 'text'] and args.file:
                # Load content from file for context/document variables
                try:
                    with open(args.file, 'r', encoding='utf-8') as f:
                        variables[var] = f.read()
                    print(f"Loaded content for '{var}' from file: {args.file}")
                except Exception as e:
                    print(f"Error reading file: {e}")
                    value = input(f"Enter value for '{var}': ")
                    variables[var] = value
            else:
                value = input(f"Enter value for '{var}': ")
                variables[var] = value
    
    # Format the prompt
    try:
        formatted_prompt = template.format(**variables)
        
        print("\nFormatted Prompt:")
        print("-" * 80)
        print(formatted_prompt)
        print("-" * 80)
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(formatted_prompt)
                print(f"Prompt saved to: {args.output}")
            except Exception as e:
                print(f"Error saving prompt to file: {e}")
        
        # Track performance if requested
        if args.track:
            optimizer = get_prompt_optimizer()
            metrics = {"test_run": 1.0}
            
            if args.metrics:
                for metric_str in args.metrics:
                    try:
                        name, value = metric_str.split('=', 1)
                        metrics[name.strip()] = float(value.strip())
                    except ValueError:
                        print(f"Invalid metric format: {metric_str}. Use name=value format.")
            
            result_id = optimizer.track_prompt_performance(
                template_name=template.name,
                prompt=formatted_prompt,
                variables=variables,
                metrics=metrics,
                llm_name=args.llm
            )
            
            print(f"Performance tracked with ID: {result_id}")
            print(f"Metrics: {metrics}")
    
    except Exception as e:
        print(f"Error formatting prompt: {e}")

def optimize_template(args):
    """Get optimization suggestions for a template."""
    library = get_prompt_library()
    optimizer = get_prompt_optimizer()
    
    # Check if template exists
    template = library.get_template(args.name)
    if not template:
        print(f"Template '{args.name}' not found.")
        return
    
    # Get suggestions
    suggestions = optimizer.suggest_improvements(args.name)
    
    print(f"\nOptimization Suggestions for '{args.name}':\n")
    
    if "suggestions" in suggestions and suggestions["suggestions"]:
        for i, suggestion in enumerate(suggestions["suggestions"], 1):
            print(f"{i}. {suggestion['message']}")
    else:
        print("No specific suggestions available for this template.")
    
    # Show analysis if requested
    if args.show_analysis and "analysis" in suggestions:
        analysis = suggestions["analysis"]
        
        print("\nPerformance Analysis:")
        
        if "results_count" in analysis and analysis["results_count"] > 0:
            print(f"Based on {analysis['results_count']} tracked usages")
            
            if "metrics_stats" in analysis:
                print("\nMetrics:")
                for metric, stats in analysis["metrics_stats"].items():
                    print(f"  {metric}: {stats['mean']:.4f} (± {stats['std']:.4f}) [min: {stats['min']:.4f}, max: {stats['max']:.4f}]")
            
            if "trends" in analysis and analysis["trends"]:
                print("\nTrends:")
                for metric, trend in analysis["trends"].items():
                    direction = "↑" if trend["direction"] == "positive" else "↓"
                    print(f"  {metric}: {direction} ({trend['change']:.4f})")
        else:
            print("No performance data available for this template.")

def export_template(args):
    """Export a template to a file."""
    library = get_prompt_library()
    
    # Check if template exists
    template = library.get_template(args.name)
    if not template:
        print(f"Template '{args.name}' not found.")
        return
    
    # Determine output format
    format_type = args.format.lower()
    if format_type not in ['json', 'yaml', 'text']:
        print(f"Unsupported format: {args.format}. Using 'json' instead.")
        format_type = 'json'
    
    # Determine output file
    output_file = args.output
    if not output_file:
        output_file = f"{template.name}.{format_type}"
    
    try:
        # Export based on format
        if format_type == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(template.to_dict(), f, indent=2)
        elif format_type == 'yaml':
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(template.to_dict(), f, default_flow_style=False)
        else:  # text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(template.template)
        
        print(f"Template '{template.name}' exported to: {output_file}")
    except Exception as e:
        print(f"Error exporting template: {e}")

def import_template(args):
    """Import a template from a file."""
    library = get_prompt_library()
    
    try:
        # Read file
        with open(args.file, 'r', encoding='utf-8') as f:
            if args.file.endswith('.json'):
                data = json.load(f)
            elif args.file.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
            else:
                # Assume it's a text file with template content
                content = f.read()
                
                # Create template data
                name = args.name or os.path.basename(args.file).split('.')[0]
                data = {
                    "template": content,
                    "name": name,
                    "description": args.description or f"Imported from {args.file}",
                    "variables": args.variables.split(",") if args.variables else None,
                    "version": "1.0"
                }
        
        # Create template
        template = PromptTemplate.from_dict(data)
        
        # Override name if provided
        if args.name:
            template.name = args.name
        
        # Check if template already exists
        existing = library.get_template(template.name)
        if existing:
            if not args.force:
                confirm = input(f"Template '{template.name}' already exists. Overwrite? (y/N): ")
                if confirm.lower() != 'y':
                    print("Import cancelled.")
                    return
            
            # Update existing template
            success = library.update_template(template.name, template)
        else:
            # Add new template
            success = library.add_template(template)
        
        if success:
            print(f"Template '{template.name}' imported successfully.")
            print(f"Variables: {', '.join(template.variables)}")
        else:
            print(f"Failed to import template '{template.name}'.")
    except Exception as e:
        print(f"Error importing template: {e}")

def restore_defaults(args):
    """Restore default templates."""
    library = get_prompt_library()
    
    # Get default templates
    defaults = DefaultPromptTemplates.get_default_templates()
    
    if not defaults:
        print("No default templates available.")
        return
    
    # Confirm restoration
    if not args.force:
        confirm = input(f"This will add/update {len(defaults)} default templates. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Restoration cancelled.")
            return
    
    # Add/update each template
    added = 0
    updated = 0
    
    for template in defaults:
        # Check if exists
        existing = library.get_template(template.name)
        
        if existing:
            # Update
            success = library.update_template(template.name, template)
            if success:
                updated += 1
        else:
            # Add
            success = library.add_template(template)
            if success:
                added += 1
    
    print(f"Default templates restored: {added} added, {updated} updated.")

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="Prompt Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List templates command
    list_parser = subparsers.add_parser("list", help="List available templates")
    
    # View template command
    view_parser = subparsers.add_parser("view", help="View a template")
    view_parser.add_argument("name", help="Template name")
    view_parser.add_argument("--performance", dest="show_performance", action="store_true", 
                           help="Show performance metrics")
    
    # Create template command
    create_parser = subparsers.add_parser("create", help="Create a new template")
    create_parser.add_argument("name", help="Template name")
    create_parser.add_argument("--description", "-d", help="Template description")
    create_parser.add_argument("--variables", "-v", help="Comma-separated list of variables")
    create_parser.add_argument("--file", "-f", help="File containing template content")
    
    # Update template command
    update_parser = subparsers.add_parser("update", help="Update an existing template")
    update_parser.add_argument("name", help="Template name")
    update_parser.add_argument("--description", "-d", help="Template description")
    update_parser.add_argument("--variables", "-v", help="Comma-separated list of variables")
    update_parser.add_argument("--file", "-f", help="File containing template content")
    update_parser.add_argument("--edit", "-e", action="store_true", 
                             help="Open template in editor")
    
    # Delete template command
    delete_parser = subparsers.add_parser("delete", help="Delete a template")
    delete_parser.add_argument("name", help="Template name")
    delete_parser.add_argument("--force", "-f", action="store_true", 
                             help="Skip confirmation")
    
    # Test template command
    test_parser = subparsers.add_parser("test", help="Test a template")
    test_parser.add_argument("name", help="Template name")
    test_parser.add_argument("--vars", "-v", action="append", 
                           help="Variable values (name=value format)")
    test_parser.add_argument("--file", "-f", help="File to load for context/document variables")
    test_parser.add_argument("--output", "-o", help="Output file for formatted prompt")
    test_parser.add_argument("--track", "-t", action="store_true", 
                           help="Track template performance")
    test_parser.add_argument("--metrics", "-m", action="append", 
                           help="Performance metrics (name=value format)")
    test_parser.add_argument("--llm", "-l", help="LLM name")
    
    # Optimize template command
    optimize_parser = subparsers.add_parser("optimize", help="Get optimization suggestions")
    optimize_parser.add_argument("name", help="Template name")
    optimize_parser.add_argument("--analysis", dest="show_analysis", action="store_true", 
                               help="Show detailed analysis")
    
    # Export template command
    export_parser = subparsers.add_parser("export", help="Export a template")
    export_parser.add_argument("name", help="Template name")
    export_parser.add_argument("--format", "-f", default="json", 
                             help="Export format (json, yaml, text)")
    export_parser.add_argument("--output", "-o", help="Output file")
    
    # Import template command
    import_parser = subparsers.add_parser("import", help="Import a template")
    import_parser.add_argument("file", help="File to import")
    import_parser.add_argument("--name", "-n", help="Template name (overrides name in file)")
    import_parser.add_argument("--description", "-d", help="Template description")
    import_parser.add_argument("--variables", "-v", help="Comma-separated list of variables")
    import_parser.add_argument("--force", "-f", action="store_true", 
                             help="Overwrite existing template")
    
    # Restore defaults command
    defaults_parser = subparsers.add_parser("defaults", help="Restore default templates")
    defaults_parser.add_argument("--force", "-f", action="store_true", 
                               help="Skip confirmation")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == "list":
        list_templates(args)
    elif args.command == "view":
        view_template(args)
    elif args.command == "create":
        create_template(args)
    elif args.command == "update":
        update_template(args)
    elif args.command == "delete":
        delete_template(args)
    elif args.command == "test":
        test_template(args)
    elif args.command == "optimize":
        optimize_template(args)
    elif args.command == "export":
        export_template(args)
    elif args.command == "import":
        import_template(args)
    elif args.command == "defaults":
        restore_defaults(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
