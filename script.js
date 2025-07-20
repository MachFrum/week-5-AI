// Page flip functionality
let currentPage = 1;
const totalPages = 3;

// Initialize page visibility
function initializePages() {
    // Hide all pages first
    for (let i = 1; i <= totalPages; i++) {
        const page = document.getElementById(`page${i}`);
        if (page) {
            page.style.display = i === 1 ? 'block' : 'none';
            if (i === 1) {
                page.classList.add('active');
            }
        }
    }
}

function flipPage(targetPage) {
    if (targetPage === currentPage) return;
    
    const currentPageEl = document.getElementById(`page${currentPage}`);
    const targetPageEl = document.getElementById(`page${targetPage}`);
    
    // Add animation classes
    currentPageEl.classList.add('flipping-out');
    currentPageEl.classList.remove('active');
    
    setTimeout(() => {
        currentPageEl.classList.remove('flipping-out');
        currentPageEl.style.display = 'none';
        
        targetPageEl.style.display = 'block';
        targetPageEl.classList.add('flipping-in');
        
        setTimeout(() => {
            targetPageEl.classList.remove('flipping-in');
            targetPageEl.classList.add('active');
            currentPage = targetPage;
            
            // Update progress bars
            updateProgress();
            
            // Trigger animations on new page
            if (targetPage === 2 || targetPage === 3) {
                animateStats();
                animateDiagrams();
            }
        }, 50);
    }, 500);
}

// Progress bar animation
function updateProgress() {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        const progress = ((currentPage - 1) / (totalPages - 1)) * 100;
        progressBar.style.width = `${progress}%`;
    }
}

// Animate statistics counters
function animateStats() {
    const statNumbers = document.querySelectorAll('.stat-number');
    
    statNumbers.forEach(stat => {
        const target = parseInt(stat.getAttribute('data-target'));
        const duration = 2000;
        const increment = target / (duration / 16);
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            stat.textContent = Math.floor(current);
        }, 16);
    });
}

// Particle Network Background
function createParticleNetwork() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const particleNetwork = document.querySelector('.particle-network');
    
    if (!particleNetwork) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    particleNetwork.appendChild(canvas);
    
    const particles = [];
    const particleCount = 50;
    const connectionDistance = 150;
    
    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 2 + 1;
        }
        
        update() {
            this.x += this.vx;
            this.y += this.vy;
            
            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }
        
        draw() {
            ctx.fillStyle = 'rgba(124, 77, 255, 0.5)';
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    // Create particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Update and draw particles
        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });
        
        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < connectionDistance) {
                    const opacity = 1 - (distance / connectionDistance);
                    ctx.strokeStyle = `rgba(124, 77, 255, ${opacity * 0.2})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

// Diagram animation
function animateDiagrams() {
    const diagramContainers = document.querySelectorAll('.diagram-container');
    diagramContainers.forEach(container => {
        container.style.opacity = '0';
        container.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            container.style.transition = 'all 0.6s ease';
            container.style.opacity = '1';
            container.style.transform = 'translateY(0)';
        }, 300);
    });
}

// Smooth scroll for article content
function enableSmoothScroll() {
    const articleBodies = document.querySelectorAll('.article-body');
    articleBodies.forEach(body => {
        body.addEventListener('wheel', (e) => {
            if (body.scrollHeight > body.clientHeight) {
                e.stopPropagation();
            }
        });
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize page visibility
    initializePages();
    
    // Create particle network
    createParticleNetwork();
    
    // Enable smooth scrolling
    enableSmoothScroll();
    
    // Set initial progress
    updateProgress();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        const canvas = document.querySelector('.particle-network canvas');
        if (canvas) {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
    });
    
    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' && currentPage < totalPages) {
            flipPage(currentPage + 1);
        } else if (e.key === 'ArrowLeft' && currentPage > 1) {
            flipPage(currentPage - 1);
        }
    });
    
    // Add touch/swipe support for mobile
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
    
    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0 && currentPage < totalPages) {
                // Swiped left, go to next page
                flipPage(currentPage + 1);
            } else if (diff < 0 && currentPage > 1) {
                // Swiped right, go to previous page
                flipPage(currentPage - 1);
            }
        }
    }
});

// Export functions for external use
window.aiHealthcare = {
    flipPage,
    animateStats,
    updateProgress
};