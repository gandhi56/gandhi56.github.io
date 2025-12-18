document.addEventListener('DOMContentLoaded', () => {
    const commitFeed = document.getElementById('commit-feed');
    const username = 'gandhi56';
    const repo = 'llvm/llvm-project';
    const apiUrl = `https://api.github.com/repos/${repo}/commits?author=${username}&per_page=6`;

    if (!commitFeed) return;

    // Show loading state
    commitFeed.innerHTML = '<p style="text-align: center; color: #aaa;">Loading latest contributions...</p>';

    fetch(apiUrl)
        .then(response => {
            if (!response.ok) {
                if (response.status === 403) {
                    throw new Error('API rate limit exceeded. Please try again later.');
                }
                throw new Error('Failed to fetch commits.');
            }
            return response.json();
        })
        .then(commits => {
            if (commits.length === 0) {
                commitFeed.innerHTML = '<p style="text-align: center;">No recent commits found.</p>';
                return;
            }

            const commitsHtml = commits.map(commit => {
                const fullMessage = commit.commit.message;
                const messageLines = fullMessage.split('\n');
                const title = messageLines[0]; // First line is the title
                
                // Get description (everything after first line, filtered for empty lines)
                const description = messageLines.slice(1).join('\n').trim();
                const descriptionHtml = description 
                    ? `<div style="margin-top: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.7); white-space: pre-wrap; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">${escapeHtml(description)}</div>` 
                    : '';

                const date = new Date(commit.commit.author.date).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                });
                const url = commit.html_url;
                const sha = commit.sha.substring(0, 7);

                return `
                    <div class="contribution-item glass-card-hover" style="display: flex; flex-direction: column; height: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; background: rgba(255,255,255,0.05); padding: 1.5rem;">
                        <div style="margin-bottom: 0.5rem; flex-grow: 1;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                                <h3 style="margin: 0; font-size: 1.1rem; color: #fff; line-height: 1.4;">
                                    <a href="${url}" target="_blank" style="text-decoration: none; color: inherit;">${escapeHtml(title)}</a>
                                </h3>
                                <span style="font-family: monospace; font-size: 0.85rem; color: var(--accent-color, #ff4d4d); background: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 4px; margin-left: 0.5rem; white-space: nowrap;">${sha}</span>
                            </div>
                            <p style="margin: 0; font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-bottom: 1rem;">
                                Committed on ${date}
                            </p>
                            ${descriptionHtml}
                        </div>
                    </div>
                `;
            }).join('');

            commitFeed.innerHTML = `<div class="commits-list">${commitsHtml}</div>`;
        })
        .catch(error => {
            console.error('Error fetching commits:', error);
            commitFeed.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <p style="color: #ff6b6b; margin-bottom: 1rem;">Unable to load dynamic commit feed.</p>
                    <a href="https://github.com/${repo}/commits?author=${username}" class="button" target="_blank">View Commits on GitHub</a>
                </div>
            `;
        });
});

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
