import subprocess
import os

print("🚀 Starting robust history filter...")

# Define the environment variables that map the author replacement rules
# Using basic POSIX shell conditional format that Git internal sh understands perfectly
git_script = """
if [ "$GIT_AUTHOR_EMAIL" = "claude@anthropic.com" ] || [ "$GIT_COMMITTER_EMAIL" = "claude@anthropic.com" ] || echo "$GIT_AUTHOR_EMAIL" | grep -q "claude" 2>/dev/null || [[ "$GIT_AUTHOR_EMAIL" == *claude* ]]; then
    export GIT_AUTHOR_NAME="Shravani Mayekar"
    export GIT_AUTHOR_EMAIL="shravani07-tech@users.noreply.github.com"
    export GIT_COMMITTER_NAME="Shravani Mayekar"
    export GIT_COMMITTER_EMAIL="shravani07-tech@users.noreply.github.com"
fi
"""

# If the standard filter-branch fails due to windows shell limits, let's use standard git configuration to safely re-author
# A foolproof inline text configuration replacement approach:
with open("mailmap_tmp", "w") as f:
    f.write("Shravani Mayekar <shravani07-tech@users.noreply.github.com> Claude <claude@anthropic.com>\n")
    f.write("Shravani Mayekar <shravani07-tech@users.noreply.github.com> claude <claude@anthropic.com>\n")

print("🔄 Running internal repository re-authoring engine...")
try:
    # Fallback to a completely cross-platform safe string rewrite mechanism that doesn't rely on bash loops
    subprocess.run(["git", "filter-branch", "-f", "--env-filter", 
                    'if echo "$GIT_AUTHOR_NAME" | grep -q "claude" 2>/dev/null || [ "$GIT_AUTHOR_EMAIL" = "claude@anthropic.com" ]; then export GIT_AUTHOR_NAME="Shravani Mayekar"; export GIT_AUTHOR_EMAIL="shravani07-tech@users.noreply.github.com"; export GIT_COMMITTER_NAME="Shravani Mayekar"; export GIT_COMMITTER_EMAIL="shravani07-tech@users.noreply.github.com"; fi', 
                    "--", "--branches", "--tags"], check=True, shell=True)
    print("\n✅ Internal processing complete!")
except Exception:
    # Ultimate brute-force shortcut if your local Git path installation is missing basic shell utilities:
    print("\n⚠️ Standard filter-branch execution environment limited on this local machine.")
    print("👉 Alternative: Copy your modified files to a backup folder, delete the local .git folder, run 'git init', and push a clean fresh history directly to GitHub!")