# extracting_FAUs
extract FAUs for multiple videos using OpenGraphAU, filter face in frame for videos, and aggregate features into one csv

# to run
- place all videos for one subject into a folder, and set that as the videoDir variable in opengraph_job.sh
- run opengraph_job.sh for the full pipeline
    - requires GPU

# how to use uv (environment management)
- install uv
    - with pypi -> [pip install uv](https://pypi.org/project/uv/)
    - w/ curl -> curl -LsSf https://astral.sh/uv/install.sh | sh 
- run 'uv init' in project folder
    - may need to do source .venv/bin/activate to activate the env
- install dependencies
    - uv pip install -r requirements.txt
    - use uv add X e.g. uv add torch to add any additional deps
