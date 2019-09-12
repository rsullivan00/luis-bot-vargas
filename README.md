# Luis Bot Vargas



## Get Card Names and Scores from Channel Fireball article

```es6
$('h1,h3').select((_, el) => el.tagName === 'H3' || el.innerText.indexOf('Limited: ') === 0).map((idx, el) => el.innerText.replace('Limited: ', '')).slice(3)
```
