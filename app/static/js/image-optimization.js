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
