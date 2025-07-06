# Version Control Guide for GNSS-InSAR Scripts

This document provides a quick guide to using Git for version control with your GNSS-InSAR alignment scripts. Version control will help you track changes, revert to previous versions if needed, and collaborate more effectively.

## Basic Git Workflow

### Getting Started

1. Run `tools\setup_git.bat` once to initialize the repository
2. After the setup is complete, you can use the commands below for day-to-day work

### Daily Workflow Commands

#### Check Status of Your Changes

```
git status
```
Shows which files have been changed, added, or deleted.

#### Review Your Changes

```
git diff
```
Shows the exact code changes you've made.

#### Save Your Changes (Committing)

```
git add .
git commit -m "Brief description of what you changed"
```

#### Create a Checkpoint (Tag) for Important Versions

```
git tag v1.0 -m "First stable version"
```

### Undoing Changes

#### Revert a File to its Last Committed Version

```
git checkout -- filename.py
```

#### Revert to a Previous Version (Tag)

```
git checkout v1.0
```

#### Create a New Branch Before Making Major Changes

```
git branch new-feature
git checkout new-feature
```
Now make your changes safely in this branch without affecting the main code.

#### Return to the Main Branch

```
git checkout main
```

## Best Practices

1. **Commit Often**: Make small, frequent commits with clear messages
2. **Create Tags**: Tag important milestones (v1.0, v1.1, etc.)
3. **Before Major Changes**: Create a branch or tag the current state
4. **Commit Messages**: Write clear descriptions of what changed and why

## Common Scenarios

### "I need to go back to yesterday's version"

```
# See commit history
git log

# Revert to a specific commit
git checkout [commit-hash]
```

### "I want to try something but don't want to risk breaking my code"

```
# Create and switch to a new branch
git checkout -b experimental

# (Make your changes and test)

# If you like the changes, merge back to main
git checkout main
git merge experimental

# If you don't like the changes, just delete the branch
git branch -D experimental
```

### "I want to backup my repository to a USB drive"

```
# Clone your repository to a USB drive
git clone . /path/to/usb/drive/scripts-backup
```

---

For more help with Git, visit [Git Documentation](https://git-scm.com/doc) or [GitHub Guides](https://guides.github.com/)
