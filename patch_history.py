import subprocess

# The bash script logic Git filter-branch expects
git_filter_script = """
if echo "$GIT_AUTHOR_EMAIL" | grep -qi "claude"; then
    export GIT_AUTHOR_NAME="Shravani Mayekar"
    export GIT_AUTHOR_EMAIL="shravani07-tech@users.noreply.github.com"
    export GIT_COMMITTER_NAME="Shravani Mayekar"
    export GIT_COMMITTER_EMAIL="shravani07-tech@users.noreply.github.com"
fi
"""

print("🚀 Rewriting commit history safely to clear out Claude's author tag...")

try:
    # Run filter-branch with clear string barriers
    subprocess.run(
        ["git", "filter-branch", "-f", "--env-filter", git_filter_script, "--", "--branches", "--tags"],
        check=True
    )
    print("\n✅ History rewrite complete! Run 'git push origin main --force' to update GitHub.")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error encountered during processing: {e}")