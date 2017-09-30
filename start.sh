#!/bin/bash

current_dir=$(pwd)
script_dir=$(dirname $0)
doc_path=$current_dir'/'$script_dir
echo 'Starting docker on http://localhost:8000 with document path '$doc_path

docker pull nguyenthuan/paddlepaddle.org:test

docker run -d --rm --name paddlepaddle.org -p 8000:8000 -e ENV=development -e SECRET_KEY="secret" -v $doc_path:/var/content nguyenthuan/paddlepaddle.org:test

sleep 1
python -mwebbrowser http://localhost:8000