const vscode = require('vscode');
const axios = require('axios');

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('Copilot RAG Extension is now active');

    let disposable = vscode.commands.registerCommand('copilot-rag.query', async function () {
        // Get query from user
        const query = await vscode.window.showInputBox({
            placeHolder: 'Enter your query',
            prompt: 'Search your local RAG system'
        });

        if (!query) return;

        // Show progress indicator
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Querying local RAG...",
            cancellable: false
        }, async (progress) => {
            try {
                // Query the local RAG API
                const response = await axios.post('http://localhost:8000/query', {
                    query,
                    top_k: 5
                });

                // Format the results
                const results = response.data.results;
                let markdown = new vscode.MarkdownString(`## Results for: ${query}\n\n`);
                
                results.forEach((result, index) => {
                    markdown.appendMarkdown(`### Result ${index + 1} (Score: ${result.score.toFixed(4)})\n\n`);
                    markdown.appendMarkdown(`Source: ${result.metadata.source}, Chunk: ${result.metadata.chunk_index}\n\n`);
                    markdown.appendMarkdown(`\`\`\`\n${result.text}\n\`\`\`\n\n`);
                });

                // Show results in a new tab
                const doc = await vscode.workspace.openTextDocument({
                    content: markdown.value,
                    language: 'markdown'
                });
                
                await vscode.window.showTextDocument(doc);
                
                // Also copy to clipboard for potential use with GitHub Copilot
                await vscode.env.clipboard.writeText(
                    results.map(r => r.text).join('\n\n')
                );
                
                vscode.window.showInformationMessage('RAG results copied to clipboard for use with GitHub Copilot');
            } catch (error) {
                vscode.window.showErrorMessage(`Error querying RAG: ${error.message}`);
            }
        });
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}