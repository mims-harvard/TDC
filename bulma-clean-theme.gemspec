# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "bulma-clean-theme"
  spec.version       = "0.6.3"
  spec.authors       = ["chrisrhymes"]
  spec.email         = ["csrhymes@gmail.com"]

  spec.summary       = "Clean and simple Jekyll theme built using Bulma"
  spec.homepage      = "https://www.github.com/chrisrhymes/bulma-clean-theme"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|_posts|blog|LICENSE|README|package|node_modules|favicon)!i) }

  spec.add_runtime_dependency "jekyll", "~> 3.8"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.11"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.2"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.5"
  spec.add_runtime_dependency "kramdown", "~> 1.17"
  spec.add_runtime_dependency "rouge", "~> 3.3"

  spec.add_development_dependency "bundler", "~> 2.0"
  spec.add_development_dependency "rake", "~> 12.0"
end
