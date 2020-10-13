source "https://rubygems.org"
ruby RUBY_VERSION

# gem "jekyll", "4.0"

# to use GitHub Pages
gem "github-pages", group: :jekyll_plugins

# If you have any plugins, put them here!
# group :jekyll_plugins do
#    gem "jekyll-feed"
#    gem "jekyll-sitemap"
#    gem "jekyll-redirect-from"
#    gem "jekyll-seo-tag"
# end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :install_if => Gem.win_platform?
