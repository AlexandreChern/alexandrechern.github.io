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

## Publishing large datasets

Large research artifacts are stored outside Git in Cloudflare R2. The website keeps only their metadata and public download URLs. Published paths are versioned and must not be overwritten.

### Configure R2

1. Create an R2 bucket and connect a public custom domain such as `data.example.com` in **R2 > bucket > Settings > Public access**.
2. Create an R2 API token with **Object Read & Write** permission restricted to that bucket.
3. Install rclone. On macOS, run `brew install rclone`.
4. Export the settings in the shell that will perform the upload:

	 ```sh
	 export R2_ACCOUNT_ID="your-cloudflare-account-id"
	 export R2_BUCKET="academic-datasets"
	 export R2_PUBLIC_BASE_URL="https://data.example.com"
	 export R2_ACCESS_KEY_ID="your-r2-access-key-id"
	 export R2_SECRET_ACCESS_KEY="your-r2-secret-access-key"
	 ```

Keep credentials out of this repository and shell history. The secret key cannot be recovered after leaving Cloudflare's token creation screen.

### Upload a release

Put the files for one immutable release in a directory outside this repository, then run:

```sh
bash scripts/publish-dataset.sh /path/to/release poisson-1d v1.0.0
```

The script uploads to `datasets/poisson-1d/v1.0.0/`, creates a `SHA256SUMS` file, and refuses to overwrite a nonempty version. If an upload is interrupted, inspect and remove only that incomplete version in R2 before retrying.

### Add it to the website

Add a record to `_data/datasets.yml` and rebuild the site:

```yaml
- title: Poisson 1D reproducibility package
	version: v1.0.0
	published: 2026-09-12
	description: Inputs, source code, and numerical results for the 1D Poisson example.
	license: MIT
	doi: 10.5281/zenodo.1234567
	doi_url: https://doi.org/10.5281/zenodo.1234567
	files:
		- name: simulation-code.zip
			format: ZIP
			size: 24 MB
			url: https://data.example.com/datasets/poisson-1d/v1.0.0/simulation-code.zip
			sha256: replace-with-the-value-from-SHA256SUMS
```

The record appears at `/datasets/`. Use a new version path for every release. For research intended to be cited permanently, publish the same release to Zenodo, Figshare, or Dryad and include its DOI; R2 is the fast distribution copy, not the archival record.
