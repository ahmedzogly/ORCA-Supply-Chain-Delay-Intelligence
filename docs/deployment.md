# Deployment Guide

The API is packaged as a standard Docker container.

## Build
`ash
docker build -t delay_intelligence_api:v1 .
`

## Run
`ash
docker run -p 8000:8000 delay_intelligence_api:v1
`

The service runs entirely locally without Kubernetes or Cloud dependencies, fulfilling the project's local-first mandate.
