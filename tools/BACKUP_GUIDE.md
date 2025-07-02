# Backup System Guide

This guide explains how to use the backup and restore scripts for your GNSS-InSAR alignment project.

## Quick Start

### Creating a Backup

1. Run `tools\create_backup.bat` before making significant changes
2. A timestamped backup of all Python files will be created in the `tools\backups` folder

### Restoring from a Backup

1. Run `tools\restore_backup.bat` when you want to revert to a previous version
2. Select the backup timestamp you want to restore from the displayed list
3. Confirm the restoration

## Recommended Backup Strategy

### When to Create Backups

- Before making significant changes to the code
- After successfully implementing a feature
- Before refactoring or restructuring code
- After fixing major bugs
- Before experimenting with different approaches

### Managing Backups

Backups are stored in the `tools\backups` folder with timestamps. You can safely delete older backups when you no longer need them.

## Using with Git

The backup system works independently of Git and is meant to provide a quick and simple way to create snapshots of your code. For more advanced version control features, see `tools\VERSION_CONTROL.md` and use Git.

## Important Note

While `restore_backup.bat` will create an automatic backup of your current state before restoring, it's still a good practice to manually create a backup before restoring if you're unsure.
