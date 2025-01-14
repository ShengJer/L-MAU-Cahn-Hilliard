import subprocess
import os
# Path to your bash script
bash_script = "./submit.sh"
if __name__ == "__main__":
    # Run the bash script
	subprocess.run(["bash", bash_script])

