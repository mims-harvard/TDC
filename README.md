# TDC Website Backend

Suppose TDC repo and tdc_website are in the same directory.

In TDC repo gh-pages branch, do
```
rm -r *
```

In this site, do
```
bundle exec jekyll serve
mv _site/* ../TDC/
```

Then, commit to both repos in remote.

