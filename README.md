# PaddlePaddle.org Documentation

### 1)  clone documentation repo and checkout branch

```
git clone https://github.com/bobateadev/doc_test.git

git checkout origin develop
# OR a specific version
git checkout origin v0.9.0
```

### 2)  Pull PaddlePaddle.org image

```
docker pull nguyenthuan/paddlepaddle.org:test
```

### 3)  Run PaddlePaddle.org with doc_test dir as the volume

```
docker run -d -p 8000:8000 -e ENV=development -e SECRET_KEY="secret" -v <PATH_OF_DOC_TEST>:/var/content nguyenthuan/paddlepaddle.org:test
```
