# Burki Design System: "Electric Slate"

This document outlines the core design principles, color palette, and typography for the Burki project. The goal is to create a consistent, professional, and memorable brand identity that reflects the project's status as a modern, high-performance developer tool.

---

## Guiding Principles

-   **Professional & Credible:** The design should inspire confidence. It's clean, structured, and avoids overly playful elements.
-   **High Contrast & Accessible:** The dark background with vibrant accents ensures excellent readability and is easy on the eyes during long sessions.
-   **Modern & Technical:** The aesthetic feels contemporary, using subtle animations and a font typically associated with developer tools.
-   **Unique & Memorable:** The "Electric Slate" theme is designed to be distinctive and recognizable.

---

## 1. Color Palette

The color palette is built on a dark, slate-like background with a single, vibrant accent color for high contrast and visual interest.

| Role                | Color          | Hex Code  | Usage                               |
| ------------------- | -------------- | --------- | ----------------------------------- |
| **Background**      | Slate Dark     | `#0D1117` | The primary background for all pages. |
| **Primary Accent**  | Electric Green | `#10B981` | CTAs, links, icons, key highlights. |
| **Accent Gradient** | Green Fade     | `#A3FFAE` | Used with the primary accent in gradients. |
| **Primary Text**    | White Smoke    | `#E6EDF3` | Headings and important text.        |
| **Secondary Text**  | Slate Gray     | `#8B949E` | Body copy and less important text.  |
| **Border Color**    | Subtle Border  | `#30363D` | Borders on cards and UI elements.   |

### CSS Variables Reference

For consistency, these colors are defined as CSS variables in the main layout templates.

```css
:root {
    --bg-color: #0D1117;
    --accent-color: #10B981;
    --accent-gradient: linear-gradient(135deg, #10B981, #A3FFAE);
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --border-color: rgba(48, 54, 61, 0.8);
}
```

---

## 2. Typography

We use a two-font system to balance personality with readability.

-   **Headings Font: Satoshi**
    -   **Weight:** 700 (Bold) & 900 (Black)
    -   **Usage:** Used for all major headings (`<h1>`, `<h2>`, etc.) to give the brand a modern, technical, and distinctive voice.
-   **Body Font: Inter**
    -   **Weight:** 400 (Regular) & 500 (Medium)
    -   **Usage:** Used for all paragraph text and UI copy. Inter is chosen for its exceptional readability at all sizes.

### Font Implementation

The fonts are imported via Fontshare in the `<head>` of the main layout files.

```html
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@700,900&f[]=inter@400,500,600&display=swap" rel="stylesheet">
```

---

## 3. UI Components & Styling

-   **Glass Morphism:** Cards, headers, and other container elements use a semi-transparent background with a `backdrop-filter` to create a sense of depth and hierarchy.
-   **Subtle Animations:** Micro-interactions on buttons (hover scale) and feature cards (translate on hover) are used to make the UI feel responsive and alive, without being distracting.
-   **Icons:** We use Font Awesome for a consistent and comprehensive set of icons throughout the application.

---

## 4. Logo & Favicon

The logo and favicon use the "Electric Slate" color palette. The design combines three core concepts:
1.  **Voice Waves:** Representing the primary function of the application.
2.  **AI Circuit Pattern:** Hinting at the underlying intelligence and technical nature.
3.  **Modern Typography:** Using the brand's primary fonts. 