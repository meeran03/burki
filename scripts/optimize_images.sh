#!/bin/bash

# Image Optimization Script for Burki Voice AI
# Converts images to WebP format for better performance and SEO

echo "🖼️  Starting image optimization for Burki..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if cwebp is installed
if ! command -v cwebp &> /dev/null; then
    echo -e "${RED}❌ cwebp not found. Installing...${NC}"
    
    # Install based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y webp
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install webp
        else
            echo -e "${RED}❌ Please install Homebrew first: https://brew.sh${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Unsupported OS. Please install webp manually.${NC}"
        exit 1
    fi
fi

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATIC_DIR="$BASE_DIR/app/static"

echo -e "${BLUE}📁 Working directory: $STATIC_DIR${NC}"

# Create optimized directory structure
OPTIMIZED_DIR="$STATIC_DIR/optimized"
mkdir -p "$OPTIMIZED_DIR"

# Statistics
TOTAL_ORIGINAL_SIZE=0
TOTAL_OPTIMIZED_SIZE=0
FILES_PROCESSED=0

# Function to convert and optimize image
optimize_image() {
    local input_file="$1"
    local quality="$2"
    local filename=$(basename "$input_file")
    local dirname=$(dirname "$input_file")
    local name="${filename%.*}"
    local ext="${filename##*.}"
    
    # Output paths
    local webp_output="$dirname/${name}.webp"
    local optimized_output="$OPTIMIZED_DIR/${name}.webp"
    
    if [[ "$ext" =~ ^(jpg|jpeg|png|gif)$ ]]; then
        echo -e "${YELLOW}⚡ Processing: $filename${NC}"
        
        # Get original size
        local original_size=$(stat -f%z "$input_file" 2>/dev/null || stat -c%s "$input_file" 2>/dev/null)
        TOTAL_ORIGINAL_SIZE=$((TOTAL_ORIGINAL_SIZE + original_size))
        
        # Convert to WebP
        cwebp -q $quality "$input_file" -o "$webp_output" -quiet
        
        # Also create optimized version
        cwebp -q $quality "$input_file" -o "$optimized_output" -quiet
        
        if [ $? -eq 0 ]; then
            # Get new size
            local new_size=$(stat -f%z "$webp_output" 2>/dev/null || stat -c%s "$webp_output" 2>/dev/null)
            TOTAL_OPTIMIZED_SIZE=$((TOTAL_OPTIMIZED_SIZE + new_size))
            
            # Calculate savings
            local savings=$((100 - (new_size * 100 / original_size)))
            
            echo -e "${GREEN}  ✅ Created WebP: ${name}.webp (${savings}% smaller)${NC}"
            FILES_PROCESSED=$((FILES_PROCESSED + 1))
        else
            echo -e "${RED}  ❌ Failed to convert: $filename${NC}"
        fi
    fi
}

# Process logo files (highest quality)
echo -e "\n${BLUE}🎨 Processing logos...${NC}"
find "$STATIC_DIR/logo" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | while read file; do
    optimize_image "$file" 95
done

# Process other images (balanced quality)
echo -e "\n${BLUE}🖼️  Processing other images...${NC}"
find "$STATIC_DIR" -path "$STATIC_DIR/logo" -prune -o -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print | while read file; do
    optimize_image "$file" 85
done

# Create responsive image helper
cat > "$STATIC_DIR/js/image-optimization.js" << 'EOF'
/**
 * Responsive Image Loading with WebP Support
 * Automatically serves WebP images when supported
 */

class ImageOptimizer {
    constructor() {
        this.supportsWebP = this.checkWebPSupport();
        this.init();
    }

    checkWebPSupport() {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        return canvas.toDataURL('image/webp').indexOf('webp') > -1;
    }

    init() {
        // Replace images with WebP versions if supported
        if (this.supportsWebP) {
            this.replaceImages();
        }

        // Setup lazy loading
        this.setupLazyLoading();
    }

    replaceImages() {
        const images = document.querySelectorAll('img[src]');
        images.forEach(img => {
            const src = img.getAttribute('src');
            if (src && (src.includes('.png') || src.includes('.jpg') || src.includes('.jpeg'))) {
                const webpSrc = src.replace(/\.(png|jpg|jpeg)$/, '.webp');
                
                // Check if WebP version exists
                this.imageExists(webpSrc).then(exists => {
                    if (exists) {
                        img.src = webpSrc;
                        console.log('Loaded WebP:', webpSrc);
                    }
                });
            }
        });
    }

    imageExists(url) {
        return new Promise(resolve => {
            const img = new Image();
            img.onload = () => resolve(true);
            img.onerror = () => resolve(false);
            img.src = url;
        });
    }

    setupLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        imageObserver.unobserve(img);
                    }
                });
            });

            images.forEach(img => imageObserver.observe(img));
        } else {
            // Fallback for older browsers
            images.forEach(img => {
                img.src = img.dataset.src;
                img.classList.remove('lazy');
            });
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageOptimizer();
});
EOF

# Generate image optimization summary
echo -e "\n${GREEN}📊 Optimization Summary:${NC}"
echo -e "Files processed: ${FILES_PROCESSED}"

if [ $FILES_PROCESSED -gt 0 ]; then
    TOTAL_SAVINGS=$((TOTAL_ORIGINAL_SIZE - TOTAL_OPTIMIZED_SIZE))
    SAVINGS_PERCENT=$((TOTAL_SAVINGS * 100 / TOTAL_ORIGINAL_SIZE))
    
    echo -e "Original size: $(numfmt --to=iec $TOTAL_ORIGINAL_SIZE)"
    echo -e "Optimized size: $(numfmt --to=iec $TOTAL_OPTIMIZED_SIZE)"
    echo -e "Total savings: $(numfmt --to=iec $TOTAL_SAVINGS) (${SAVINGS_PERCENT}%)"
    
    echo -e "\n${GREEN}✅ Image optimization complete!${NC}"
    echo -e "${BLUE}💡 Add the image optimization script to your HTML:${NC}"
    echo -e "<script src=\"/static/js/image-optimization.js\"></script>"
else
    echo -e "${YELLOW}⚠️  No images found to optimize${NC}"
fi

echo -e "\n${BLUE}🚀 Next steps for maximum SEO impact:${NC}"
echo -e "1. Update your templates to use WebP images"
echo -e "2. Add lazy loading attributes: data-src instead of src"
echo -e "3. Include proper alt text for all images"
echo -e "4. Use responsive image sizes with srcset"

# Create .htaccess for Apache servers
cat > "$STATIC_DIR/.htaccess" << 'EOF'
# WebP Image Serving
<IfModule mod_rewrite.c>
    RewriteEngine On
    
    # Check if browser supports WebP
    RewriteCond %{HTTP_ACCEPT} image/webp
    # Check if WebP version exists
    RewriteCond %{REQUEST_FILENAME}.webp -f
    # Serve WebP version
    RewriteRule ^(.+)\.(jpg|jpeg|png)$ $1.$2.webp [T=image/webp,E=accept:1]
</IfModule>

# Add Vary header for proper caching
<IfModule mod_headers.c>
    Header append Vary Accept env=REDIRECT_accept
</IfModule>

# Cache static assets
<IfModule mod_expires.c>
    ExpiresActive on
    ExpiresByType image/webp "access plus 1 year"
    ExpiresByType image/png "access plus 1 year"
    ExpiresByType image/jpg "access plus 1 year"
    ExpiresByType image/jpeg "access plus 1 year"
    ExpiresByType image/svg+xml "access plus 1 year"
</IfModule>
EOF

echo -e "\n${GREEN}🎉 All optimizations complete! Your images are now SEO-ready.${NC}" 