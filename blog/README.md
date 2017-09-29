# Paddle Blog

## Run locally

1. [Install Docker](https://docs.docker.com/docker-for-mac/install/)

1. Run the following command inside the repo directory:
    ```bash
    docker run --rm -v `pwd`:/data -it -p 4000:4000 jekyll/jekyll:3.5 bash -c 'cd /data && jekyll serve --watch'
    ```

1. Open browser and visit http://localhost:4000
