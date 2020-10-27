---
layout: post
title: Creating a docs site with Bulma Clean Theme
description: How to create a docs site for your project with Bulma Clean Theme 
date: 2020-05-08 09:00:07
hero_image: https://www.csrhymes.com/img/example-docs-page.jpg
hero_height: is-large
hero_darken: true
image: https://www.csrhymes.com/img/example-docs-page.jpg
tags: bulma-clean-theme jekyll docs
canonical_url: https://www.csrhymes.com/2020/05/08/creating-a-docs-site-with-bulma-clean-theme.html
---

I created Bulma Clean Theme as a theme for my own website and decided to open source it so others could use it as well. One of the key things I wanted to do was to create a theme that worked with GitHub Pages, which also means that you can also use it as a docs site for your project. 

## GitHub Pages Configuration

GitHub pages allows you to create a website for your project with free hosting. Go to your repo on GitHub, then click Settings, then scroll down to the GitHub Pages section. You have the option to create a site from the root of your master branch of from the /docs directory in your master branch. For this example, we are going to use the /docs directory. 

Don't change this setting just yet as if you don't have a docs directory there will be nothing there to publish. 

## Creating the docs directory

Clone your git repo to a local directory, let's say `~/code/my-project` for this example. The below assumes you don't yet have a docs directory and you have [jekyll installed](https://jekyllrb.com/docs/installation/). If you do already have a docs directory you will have to rename it to something else. 

Create a new jekyll installation in the docs directory, ensuring you replace your username and project name in the below example.

```bash
git clone https://github.com/username/my-project.git ~/code/my-project
cd ~/code/my-project
jekyll new docs
```

You should now have a base install of Jekyll in your freshly created docs directory. 

## Configuring the theme

1. Replace everything in the Gemfile with the following
```
source 'https://rubygems.org'
gem "bulma-clean-theme",  '0.7.2'
gem 'github-pages', group: :jekyll_plugins
```

2. Open the `_config.yml` and comment out or delete the line `theme: minima` and replace it with `remote_theme: chrisrhymes/bulma-clean-theme`, then add `github-pages` to the list of plugins. Update the baseurl to your GitHub repo name, in this example we are using `my-project` as the repo name
```yaml
#theme: minima
remote_theme: chrisrhymes/bulma-clean-theme
baseurl: "/my-project"
plugins:
- github-pages
```

3. Open the `index.md` file and update the front matter so the layout is page and then add a title
```yaml
layout: page
title: Docs for My Project
```

4. Run `bundle install` and then `bundle exec jekyll serve`

5. Visit `http://localhost:4000/my-project/` to view your new docs page.

## Menu

To create a menu on the left on your docs page you need to create a new yaml file in _data directory, such as `menu.yaml` and then use the below format, where the label will be the menu title and the items are the menu items. Each menu item can have a list of sub menu items if needed.

```yaml
- label: Example Menu
  items:
    - name: Menu item
      link: /link/
      items:
        - name: Sub menu item 
          link: /sub-menu-item/
```

## Table of contents

If you would like auto generated table of contents for your docs page then add `toc: true` to the page's front matter. The table of contents works for markdown pages and loops through the heading 2s and heading 3s in the markdown and then auto generates the contents.

## GitHub Sponsors

If you want to link to your GitHub sponsors profile then add `gh_sponsor` with your username to the `_config.yml` file.

```
gh_sponsor: chrisrhymes
```

## Making the docs page live

Once you have finished creating your docs page you can commit your changes and push everything up to GitHub. Go back to the GitHub settings page and scroll back down to the GitHub Pages section. Now we can update the setting to use the Master branch /docs folder and then GitHub will build your new docs page. 

## Want to see an example?

I recently updated one of my own packages to use Bulma Clean Theme to power the docs page. Check out the docs for [Bulma Block List](https://www.csrhymes.com/bulma-block-list) as an example. 