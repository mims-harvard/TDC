---
layout: page
title: Page With Video
description: A page with an embedded YouTube video
menubar: example_menu
show_sidebar: false
---

This is an example page with an embedded YouTube video. 

{% include youtube.html video="iRuJufELrWo" %}

To embed the video, use an include where you want the video to appear and then pass in the YouTube id as the video variable. 

{% raw %}
```liquid
{% include youtube.html video="videoid" %}
```
{% endraw %}