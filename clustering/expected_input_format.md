# Bluesky Documentation

## Kinds that are possible

``` python
["commit", "identity", "account"]
```

We only care about `commit`.

## Commits we care about

``` python
[
"app.bsky.feed.like",
"app.bsky.graph.follow",
"app.bsky.feed.post",
"app.bsky.feed.repost",
"app.bsky.actor.profile",
"app.bsky.graph.block",
"app.bsky.feed.threadgate" -- https://docs.bsky.app/docs/tutorials/thread-gates 
"app.bsky.feed.postgate",
"app.bsky.graph.listitem",
"app.bsky.graph.listblock",
"app.bsky.graph.list", -- https://docs.bsky.app/docs/tutorials/user-lists
"app.bsky.graph.starterpack",
"app.bsky.feed.generator",
"app.bsky.actor.preferences" ]
```

### Mapping

| collection | operation | interaction | notes | 
| :-- | :--  | :-- | :-- | 
| `app.bsky.feed.like`  | "create" | like | |
| `app.bsky.feed.like`  | "delete" | unlike | |
| `app.bsky.graph.follow` | "create" | follow | |
| `app.bsky.graph.follow` | "delete" | unfollow | |
| `app.bsky.feed.post` | "create" | create new post | |
| `app.bsky.feed.post` | "update" | update existing post | what caused this update? | 
| `app.bsky.feed.post` | "delete" | delete new post |  |
| `app.bsky.feed.repost` | "create" | create repost |  |
| `app.bsky.feed.repost` | "delete" | delete repost |  |
| `app.bsky.actor.profile` | "create" / "update" | create / update profile  | | 
| `app.bsky.graph.block` | "create" | block | |
| `app.bsky.graph.block` | "delete" | unblock | |


