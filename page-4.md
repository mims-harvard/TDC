---
title: Page with Tabs
subtitle: Demo page with tabs
layout: page
show_sidebar: false
tabs: example_tabs
menubar: example_menu
---

## Displaying tabs

The tabs gets its content from a data file in your site's `_data` directory. Simply set the name of your data file in the page's menubar setting in the frontmatter. 

```yml
title: Page with tabs
subtitle: Demo page with tabs
layout: page
show_sidebar: false
menubar: example_menu
tabs: example_tabs
```

Tabs can be used in conjunction with menubar and/or sidebar if you wish. 

## Creating a tabs data file

Create a data file in the _data directory and use the following format (if using yml)

```yml
alignment: is-left
style: is-boxed
size: is-large
items:
  - name: Tabs
    link: /page-4/
    icon: fa-smile-wink
  - name: Sidebar
    link: /page-1/
    icon: fa-square
  - name: No Sidebar
    link: /page-2/
    icon: fa-ellipsis-v
  - name: Menubar
    link: /page-3/
    icon: fa-bars
```

## Settings

You can control the alignment, style and size of the tabs by using the relevant [Bulma tabs classes](https://bulma.io/documentation/components/tabs/). 

## Active Tab Highlighting

It will automatically mark the active tab based on the current page.

## Icons

You can add icons to your tab by passing in the [Font Awesome icon class](https://fontawesome.com/icons?d=gallery).

If you don't wish to show icons then simply omit the option from your yaml file.