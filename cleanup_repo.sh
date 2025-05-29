#!/bin/bash
# Script to clean up large files from Git history and push to GitHub

echo "Creating backup branch before cleaning..."
git branch backup_before_cleanup

echo "Removing large model cache files from Git history..."
git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch model_cache/" \
  --prune-empty --tag-name-filter cat -- --all

echo "Removing Git garbage and optimizing repository..."
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now

echo "Forcing push to remote repository..."
git push origin --force --all
git push origin --force --tags

echo "Cleanup complete. Repository has been cleaned and pushed to GitHub."
