---
layout: page
title: Sponsors Page
subtitle: An example sponsors page
sponsors: example_sponsors
show_sidebar: false
---

## Sponsor link in navbar

If you have a GitHub sponsors account set up, you can add your username to `gh_sponsor` in the `_config.yml` file and it will display a link to your profile on the right of the navbar.

```yaml
gh_sponsor: chrisrhymes
```

## Creating a Sponsors Datafile

If you would like to create a page to thank your sponsors then create a data file, such as my_sponsors.yml file with the following structure:

```yaml
- tier_name: Platinum Sponsors
  size: large
  description: |-
    This is the description for the Platinum Tier
  sponsors:
    - name: Dave McDave
      profile: https://github.com/
    - name: Sarah Lee-Cheesecake
      profile: https://github.com/
- tier_name: Gold Sponsors
  description: |-
    This is the description for the Gold Tier
  sponsors:
    - name: Dave McDave
      profile: https://github.com/
```

The `tier_name` and `description` are required. The `size` is not required, but can be overwritten to 'large' or 'small' to increase or decrease the size of the box and the text size.
 
The sponsors require a name, but not a profile link.

## Displaying the Sponsors

To display the sponsors on your page, set the sponsors to the filename without the extension in the page's front matter

```yaml
layout: page
title: My Sponsors Page
sponsors: my_sponsors
```