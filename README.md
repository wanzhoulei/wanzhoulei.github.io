# wanzhoulei.github.io

Personal website of Wanzhou Lei, served by GitHub Pages at https://wanzhoulei.github.io.

Plain HTML and CSS with no build step. Everything is in `index.html` and `style.css`.

## Preview locally

```bash
cd ~/Desktop/wanzhoulei.github.io
python3 -m http.server 8000
# open http://localhost:8000
```

## Common updates

| Change | Where |
|--------|-------|
| News item | `index.html`, section `id="news"`: copy one `<li>`, newest first |
| Publication | `index.html`, section `id="publications"`: copy one `<li>` |
| Selected research | `index.html`, section `id="research"`: copy one `<article class="project">`; image goes in `assets/research/` (2:1 aspect) |
| CV | overwrite `assets/Wanzhou_Lei_CV.pdf` (keep the file name so links stay valid) |
| Photo | overwrite `assets/photo.jpg` (5:6 portrait, strip EXIF/GPS first) |
| Colors and fonts | variables at the top of `style.css` |
| "Last updated" | footer at the bottom of `index.html` |

Email and phone are stored reversed in `data-` attributes and assembled by the script at the bottom of `index.html`, so plain scrapers do not see them.

## Photo gallery

`gallery.html` builds its sections from `assets/gallery/photos.js`: one section per `region` (`china`, `neotropics`, `other`), photos in list order, and `"featured": true` for a large 2x2 tile. Add a photo with:

```bash
python3 tools/add_photos.py ~/Pictures/IMG_1234.jpg --region china --species "Aldania imitans" \
    --cn "仿斑伞蛱蝶" --place "Mingyong, Yunnan" --date "Jun 2026" [--note underside] [--featured]
```

The script resizes the photo (2000 px long edge plus an 800 px thumbnail), converts it to sRGB, strips all metadata including GPS, and adds its entry at the top of its section. To fix a caption or reorder photos, edit `photos.js` directly. To remove a photo, delete its line there and its two files (`assets/gallery/NAME.jpg` and `assets/gallery/thumbs/NAME.jpg`). The banner is `assets/gallery/hero.jpg` (3.2:1). One photo can be linked directly as `gallery.html#photo-3`.

The layout is gap-free when each section's featured count leaves the rest a multiple of 4 (for example 13 photos with 1 featured, 15 with 3).

## Notes

The Notes section (`notes/`, linked from the "Notes" button in the header) is generated from Markdown sources in `notes/src/`. Edit a source file, then rebuild:

```bash
python3 tools/build_notes.py   # needs pandoc (brew install pandoc)
```

This writes `notes/<name>.html` for every `notes/src/<name>.md` and regenerates `notes/index.html`. Commit both the sources and the generated pages. Each source file starts with front matter (`title`, `subtitle`, `order`, `updated`); math uses `$...$` and `$$...$$` (rendered by KaTeX in the browser). Special blocks:

| Markdown | Renders as |
|----------|-----------|
| `::: qa` with the question as the first paragraph | a collapsible practice question with its answer |
| `::: {.qa asked="Company, source, year"}` | the same, with a badge marking a question reported in a real interview |
| `::: key` / `::: pitfall` | a blue "Key idea" or orange "Common pitfall" box |

Styles are in `notes/notes.css`; the page template is `tools/notes/note.template.html`; the block rules are in `tools/notes/notes.lua`.

## Publish

```bash
git add -A
git commit -m "Update site"
git push
```

GitHub Pages rebuilds within a minute or two.

## Do not publish

`materials/` (paper PDFs, original photo) is listed in `.gitignore` and must stay out of the repository. Paper PDFs are never hosted here; the site links only to public arXiv pages.
