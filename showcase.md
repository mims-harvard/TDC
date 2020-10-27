---
title: Showcase
subtitle: An example showcase page
description: An example showcase page to help you easily display your work
layout: page
showcase: showcase_example
show_sidebar: false
---

## Creating A Showcase Datafile

Create a datafile in your sites `_data` directory in the following format. Subtitle, features and tags are not required. 

The description text accepts markdown and is run through the markdownify filter on the page.

The image_ratio will default to is-16by9 if it is not defined and accepts the [Bulma image](https://bulma.io/documentation/elements/image/) classes.

To display GitHub Stars, Forks and Watchers badges add your GitHub user and repo name to the github setting, such as `chrisrhymes/bulma-clean-theme`

To change the default styles of the features, set `features_styles`. This uses the styles from [bulma-block-list](https://www.csrhymes.com/bulma-block-list/) npm package.

```yaml
intro: |-
  This is some introduction text for the showcases.
  
  ## Example Heading
  It can convert markdown format

items:
  - title: Example showcase item
    subtitle: Example subtitle
    description: |-
      This is the example description for this item that you are showcasing and has an image, title, description, tags and a link.
    features:
      - This is a feature
      - This is a feature
    features_styles: is-centered is-outlined is-primary
    image: https://via.placeholder.com/1024x788
    image_ratio: is-16by9
    link: http://www.example.com
    link_text: View example
    tags: PHP,CSS,JavaScript
    github: user/repo-name
```

## Displaying the Showcase

Set the showcase in the page's front matter to be the name of the showcase data file without the extension. This gives you the ability to create multiple showcases to be used on different pages. 

```yaml
title: Showcase
subtitle: An example showcase page
layout: page
showcase: showcase_example
show_sidebar: false
```



