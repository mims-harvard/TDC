---
layout: post
title:  "Getting started with Bulma Clean Theme for Jekyll"
date:   2019-02-09 20:00:07
categories: development
description: "Getting started with Bulma Clean Theme for Jekyll and using it with GitHub Pages"
image: 'https://www.csrhymes.com/img/bulma-clean-theme.jpg'
published: true
canonical_url: https://www.csrhymes.com/development/2019/02/09/getting-started-with-bulma-clean-theme.html
---

I have made a [theme for Jekyll](https://rubygems.org/gems/bulma-clean-theme) which is based on the Bulma frontend framework. It might look familiar, because I am also using it for this site. If you like the look of this theme then I thought I would write a little blog post about how to use it for your own site and how you can use it with GitHub Pages. 

## Getting started

First things first, you need a local instance of Jekyll running on your computer. I'm assuming you are familiar with Jekyll and have everything you need installed. If this is not the case, check out the documentation on the [Jekyll website](https://jekyllrb.com/docs/). For this example, lets call the site myblog.

Create a new installation of Jekyll, then go into the myblog directory:
```bash
jekyll new myblog
cd myblog
```

Then add the theme to the Gemfile:
```
gem "bulma-clean-theme"
```

Then add the theme to your _config.yml:
```yaml
theme: bulma-clean-theme
```

Then run bundle to install everything
```bash
bundle
```

You can then preview your site by running the usual command
```bash
bundle exec jekyll serve
```

## Creating pages

A page can either be a html or a markdown file, as long as you set the frontmatter. There are a few settings that pages can use to customise the default theme a bit further if you so choose. 

```yaml
---
layout: page
title: Page Title
subtitle: Page Subtitle
image: /path/to/image.jpg
description: The pages meta description
hero_image: /path/to/hero-image.jpg
hero_height: is-fullheight
---
```

If you don't set a subtitle, image, hero_image or hero_height then defaults will be used or in the case of the hero_image, no image will be used on the hero. 

The theme uses the jekyll-seo-tag plugin so it will use the information you provide in the frontmatter to auto populate the meta tags and open graph tags. 

## Posts

Posts are created as per the standard Jekyll method, in a _posts directory, with each post named YYYY-MM-DD-name-of-post.markdown. For the post frontmatter you can use the same settings as a page, except you need to change the layout to post and add date and author settings. 

## Blog page

For the blog homepage, create a blog.html page with `layout: blog` and use the other settings from a normal page. The theme is set up to use jekyll-paginate so you just need to add pagination options to your _config.yml

```yml
# Pagination settings
paginate: 5
paginate_path: "/blog/page:num"
```

## Site defaults

If you don't want to set each hero_image individually, then you can set default values in your _config.yml. The below example sets a default author, layout and hero image for every post. It also turns on the side bar on the right of post pages, which will display links to your latest posts. 

```yml
defaults:
  -
    scope:
      path: ""
      type: "posts"
    values:
      author: "Author Name"
      layout: post
      hero_image: /path/to/hero-image.jpg
      show_sidebar: true
```

## Styles

The theme uses Bulma frontend framework which provides a wide range of sass variable customisations. If you want to overwrite any of the standard variables, such as the primary theme colour, then set a sass variable in a new file in assets/css/app.scss before importing the main stylesheet.

```scss
---
---
$primary: #333333;
// Import Main CSS file from theme
@import "main";
```

You can also add any of your own custom css to this file if you want to. 

### Navigation

Once you have created posts and pages, you will need to create a way for visitors to access them. The theme makes use of the Bulma navbar, which is configured through a simple yaml file. All you need to do is create a navigation.yml file in `_data` directory with the following format with the pages you want to include in the top navigation. You can now also add items to a dropdown menu.

```yaml
- name: Page 1
  link: page-1
- name: Blog
  link: blog
  dropdown: 
    - name: Page 2
      link: page-2
```

Bulma is pretty handy in the way it converts the same HTML into a mobile / tablet friendly navigation, so you only need to set the menu up once for all screen sizes. 

## Using bulma-clean-theme with Github Pages

For the site to work with Github Pages, all you need to do is update the _config.yml so it uses remote_theme instead of theme and update it to chrisrhymes/bulma-clean-theme so it knows which GitHub repo to pull the theme from when it builds your site. 

```yaml
#theme: bulma-clean-theme
remote_theme: chrisrhymes/bulma-clean-theme
```

And then push up your changes to Github as normal. 

There seems to be an issue where Jekyll doesn't work locally with remote_theme, so when working locally you have to add theme back in, but remember to comment theme out again before pushing back up to GitHub or you will probably get a build warning email.

## I hope you're not feeling overwhelmed

It may seem like there is a lot to do to get started, but really it shouldn't take very long to get a site up and running. All the options are there just in case you want to further customise the theme to be more personal to you, but you can just use the basic minimal settings to get yourself up and running. 

## Feedback and Issues

If you have any feedback, ideas or issues with how the theme could be improved, then please create an issue on the theme's GitHub page and I will try and take a look into it as soon as I can. The theme is still quite new and I have quite a few ideas for future enhancements, so I will write a new blog post on this site when any new features become available. 
