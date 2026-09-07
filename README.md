# Chern's Homepage

This is Alexandre's Homepage

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
