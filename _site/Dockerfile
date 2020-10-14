ARG JEKYLL_BASEURL=''

####################################

FROM ruby:alpine as builder

RUN apk add --no-cache make build-base
RUN gem install bundler

WORKDIR /jekyll
ADD Gemfile Gemfile.lock ./
RUN bundle install

ADD . .
ARG JEKYLL_BASEURL
RUN bundle exec jekyll build --baseurl $JEKYLL_BASEURL

####################################

FROM nginx:alpine

ARG JEKYLL_BASEURL
COPY --from=builder /jekyll/_site /usr/share/nginx/html/$JEKYLL_BASEURL
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
