# Chern's Homepage

This is Alexandre's Homepage

## Local preview

Run the Jekyll development server from the project directory:

```sh
./preview.sh
```

The site is available at `http://127.0.0.1:4001`. To use a different port, pass it as an argument, for example `./preview.sh 4002`.

## Publishing a blog post

1. Add a Markdown file to `_posts` named `YYYY-MM-DD-post-title.md`.
2. Start it with this front matter:

	 ```yaml
	 ---
	 layout: post
	 title: "Post title"
	 description: "A short summary shown on the blog page."
	 tags:
		 - HPC
		 - Numerical Methods
	 ---
	 ```

3. Write the article below the front matter and push it to GitHub. Jekyll publishes it automatically at `/blog/post-title/` and updates the tag filters on `/blog/`.
