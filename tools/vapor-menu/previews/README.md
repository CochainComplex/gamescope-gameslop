# gamescope-gameslop styles and HUD layouts

These images are generated from the actual menu artwork and runtime HUD
formatter with illustrative values. They are CPU layout previews, not game
captures or measurements of framegen image quality. The backgrounds behind
the HUD make its transparency visible.

| Selectable menu style | Preview |
|---|---|
| **Quiet** — current default; faded cyan/mauve on a dark translucent panel. | [quiet.png](quiet.png) |
| **Mallsoft** — warm stone, washed teal, dusty rose and an empty glass/column motif in the header. | [mallsoft.png](mallsoft.png) |
| **CGA** — cyan at the top, magenta at the bottom, mixed through chunky ordered pixels; black/white text. | [cga.png](cga.png) |

Choose **Menu Style** with Left/Right or Enter. It previews immediately.
**Save Preset** remembers the style for that game's preset. Appearance controls
are visible in the right column; all settings fit together without scrolling.
Tab switches groups. HUD BG Opacity (Auto or 0–100%) changes only the
backing; HUD Text Size selects Auto or 1–6×. HUD and gameplay settings apply
on the next launch. Quiet and Mallsoft share the muted HUD palette. The compact HUD includes an illustrative checkout hash; live builds use their
compiled Git version. CGA selects its dithered HUD
background on the next launch; the companion style previews immediately.

| HUD mode | Preview |
|---|---|
| Off — scene visible with no HUD. | [hud-off.png](hud-off.png) |
| Compact — labelled status and the original small deadline/jitter graphs. | [hud-compact.png](hud-compact.png) |
| Detailed — full device, rates, mode, recovery, pacing and learning diagnostics. | [hud-detailed.png](hud-detailed.png) |
| CGA compact — translucent cyan/magenta pixel fade, outlined text and small graphs. | [hud-cga-compact.png](hud-cga-compact.png) |
| CGA detailed — the same CGA background with full diagnostics. | [hud-cga-detailed.png](hud-cga-detailed.png) |
| CGA clear — 0% background opacity; lettering stays visible. | [hud-cga-clear.png](hud-cga-clear.png) |
| CGA solid — 100% background opacity. | [hud-cga-opaque.png](hud-cga-opaque.png) |
| CGA larger text — 3× text size with spaced rows. | [hud-cga-large.png](hud-cga-large.png) |

Regenerate all previews from the repository root (Pillow required):

```sh
python3 tools/vapor-menu/generate-previews.py
```

The alternate style is a design interpretation of vaporwave's ambivalent
commercial nostalgia and idealized, artificial shopping interiors. Research:
[Bandcamp Daily's iconography survey](https://daily.bandcamp.com/features/vaporwave-iconography-column)
and [Cambridge's *Vaporwave Aesthetics: Internet Nostalgia and the Utopian Impulse*](https://www.repository.cam.ac.uk/items/a33d041f-0e86-47a2-9763-3b37a147d0aa).

The CGA interpretation references [Alley Cat](https://classicreload.com/alley-cat.html)
and [CGA palette/dithering examples](https://www.pixelatedarcade.com/news/even-more-tales-of-cga-colors).
It uses the canonical high-intensity cyan/magenta/black/white palette, not an
emulation of a particular monitor or composite-video artifact.
