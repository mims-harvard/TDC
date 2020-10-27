# bulma-block-list

![npm](https://img.shields.io/npm/dw/bulma-block-list)

A simple scss package extending Bulma with block style list elements

* `npm i -S bulma-block-list`
* `@import "node_modules/bulma-block-list/src/block-list";` after importing [Bulma](https://bulma.io/documentation/customize/with-node-sass/#3-create-a-sass-file). 
* Create a ul and give it the class of `block-list`
* Add any modifier classes to change the styles as needed

```html
<ul class="block-list is-small is-outlined is-success is-centered">
    <li>Item one</li>
    <li>Item two</li>
    <li>Item three</li>
</ul>
```

## List Item Overrides

In v0.3 you can override the list item styles.

```html
<ul class="block-list is-small">
    <li>Item one</li>
    <li class="is-primary is-outlined is-large">Item two</li>
    <li>Item three</li>
</ul>
```

For examples see [csrhymes.com/bulma-block-list](https://www.csrhymes.com/bulma-block-list)