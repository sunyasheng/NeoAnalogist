from datetime import datetime
import os
import subprocess
import uuid


class TaskRollback:
    """A class that handles saving snapshots of a repository state.
    
    This class provides functionality to save the current state of a repository
    using Git and DVC for version control.
    """
    
    def save_snapshot(self, repo_dir: str) -> str:
        """Save a snapshot of the current repository state.
        
        Args:
            repo_dir (str): Path to the repository directory
            
        Returns:
            str: The commit tag of the created snapshot
        """
        # Create timestamp-based tag with random string to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        tag = f"snapshot_{timestamp}_{random_str}"

        # Save current directory
        current_dir = os.getcwd()
        try:
            # Change to repository directory
            os.chdir(repo_dir)
            print(f"Changed to directory: {os.getcwd()}")
            
            # Initialize Git repository if not already initialized
            if not os.path.exists('.git'):
                print("Initializing Git repository...")
                self._run('git init')
                self._run('git config --local user.name "GPT Scientist"')
                self._run('git config --local user.email "gpt-scientist@example.com"')
            
            # Try to initialize DVC if data directory exists and DVC is available
            if os.path.exists('data') and not os.path.exists('.dvc'):
                print("Found data directory, trying to initialize DVC...")
                try:
                    self._run('dvc init --no-scm')
                    print("DVC initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize DVC: {e}")
                    print("DVC may not be installed. Continuing with Git-only snapshot...")
            
            # Add data files to DVC if data directory exists and DVC is initialized
            if os.path.exists('data') and os.path.exists('.dvc'):
                print("Adding data files to DVC...")
                try:
                    self._run('dvc add data')
                    print("Successfully added data to DVC")
                except Exception as e:
                    print(f"Warning: Could not add data to DVC: {e}")
                    print("Continuing with Git-only snapshot...")
            else:
                if os.path.exists('data'):
                    print("Data directory found but DVC not available, using Git-only snapshot")
                else:
                    print("No data directory found, using Git-only snapshot")
            
            # Commit changes
            print("Committing changes...")
            self._run('git add .')
            commit_output = self._run('git commit -m "Create snapshot"')
            print(f"Commit output: {commit_output}")
            
            # Create tag
            print("Creating tag...")
            tag_output = self._run(f'git tag {tag}')
            print(f"Tag output: {tag_output}")
            
            # Verify tag was created
            tag_list = self._run(f'git tag -l "{tag}"')
            if not tag_list:
                raise Exception(f"Failed to create tag {tag}")
            
            return tag
            
        except Exception as e:
            print(f"Error in save_snapshot: {str(e)}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Restore original directory
            os.chdir(current_dir)

    def rollback(self, repo_dir: str, tag: str) -> None:
        """Roll back the repository to a specific snapshot.
        
        Args:
            repo_dir (str): Path to the repository directory
            tag (str): The tag of the snapshot to roll back to
            
        Raises:
            Exception: If the rollback operation fails
        """
        # Save current directory
        current_dir = os.getcwd()
        try:
            # Change to repository directory
            os.chdir(repo_dir)
            print(f"Changed to directory: {os.getcwd()}")
            
            # Check if tag exists
            result = subprocess.run(f'git tag -l "{tag}"', shell=True, capture_output=True, text=True)
            if not result.stdout.strip():
                raise Exception(f"Tag {tag} does not exist")
            
            # Reset to the tagged commit
            print(f"Rolling back to tag: {tag}")
            self._run(f'git reset --hard {tag}')
            
            # If DVC is initialized, update DVC files
            if os.path.exists('.dvc'):
                print("Updating DVC files...")
                self._run('dvc checkout')
            
        except Exception as e:
            print(f"Error in rollback: {str(e)}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Restore original directory
            os.chdir(current_dir)

    def _run(self, cmd: str) -> str:
        """Run a shell command and return its output.
        
        Args:
            cmd: The command to run
            
        Returns:
            str: The command output
            
        Raises:
            Exception: If the command fails
        """
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
