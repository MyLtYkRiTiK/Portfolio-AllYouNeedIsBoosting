#!/bin/bash
docker build -t boostingflaskimg -f Dockerfile .
docker-compose up -d