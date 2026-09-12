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

## Enabling blog discussions

Blog posts support comments through [Giscus](https://giscus.app), backed by GitHub Discussions. To enable them:

1. In the repository's **Settings > General > Features**, enable **Discussions**.
2. Install the [Giscus GitHub App](https://github.com/apps/giscus) for this repository.
3. Visit [giscus.app](https://giscus.app), enter `AlexandreChern/alexandrechern.github.io`, and choose the **Announcements** discussion category.
4. Copy the generated `data-repo-id` and `data-category-id` values into `giscus.repo_id` and `giscus.category_id` in `_config.yml`.
5. Set `giscus.enabled` to `true`, then rebuild and deploy the site.

Giscus maps each discussion to the blog post's pathname. Readers sign in with GitHub before creating comments or reactions. To disable comments for the whole site, set `giscus.enabled` back to `false`.
