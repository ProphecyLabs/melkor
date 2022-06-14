# syntax=docker/dockerfile:1

FROM python:slim-buster

MAINTAINER olga.kravchenko@prophecylabs.com

WORKDIR /app

COPY . .
RUN pip install --upgrade pip
RUN pip install -e .

EXPOSE 8889