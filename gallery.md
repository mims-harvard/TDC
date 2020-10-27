---
layout: page
title: Image Gallery
subtitle: An example image gallery page
description: A simple image gallery page 
gallery: example_gallery
show_sidebar: false
---

This is a page displaying a simple image gallery.

## Create an image gallery data file

Start by creating a gallery data file using the below format, for example `my_gallery.yml`. The image (link) will display on the page and when you click on the image, it will display the large_link image in a modal window. 

```yaml
- title: Image Gallery Title
  images:
    - link: https://via.placeholder.com/800x450
      large_link: https://via.placeholder.com/1200x675
      alt: The alt text for the image
      description: |-
        The image description can be written in **markdown** if required
      ratio: is-16by9
    - link: https://via.placeholder.com/800x600
      large_link: https://via.placeholder.com/1200x900
      alt: The alt text for the image
      description: The image description
      ratio: is-4by3
```

* If a ratio is not provided it will default to 16 by 9. Use [Bulma image](https://bulma.io/documentation/elements/image/) classes to define the image ratio required. 
* The description can be plain text or it can be markdown if required. 
* The alt will be used as the images alt text.

## Multiple galleries

If you want multiple image galleries on the same page then create additional sections in your yaml file.

```yaml
- title: First Image Gallery Title
  images:
    - link: https://via.placeholder.com/800x450
      alt: The alt text for the image
      description: |-
        The image description can be written in **markdown** if required
      ratio: is-16by9
    - link: https://via.placeholder.com/800x600
      alt: The alt text for the image
      description: The image description
      ratio: is-4by3

- title: Second Image Gallery Title
  images:
    - link: https://via.placeholder.com/800x450
      alt: The alt text for the image
      description: |-
        The image description can be written in **markdown** if required
      ratio: is-16by9
    - link: https://via.placeholder.com/800x600
      alt: The alt text for the image
      description: The image description
      ratio: is-4by3
```

## Displaying the gallery

In your pages front matter add a gallery with the datafiles filename without the extension.

```yaml
layout: page
title: My Image Gallery
gallery: my_gallery
```