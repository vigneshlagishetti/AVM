/**
 * IDS MANET — Client-side JavaScript
 * Handles form validation, animations, and interactive elements
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    initTooltips();
    
    // Form validation
    initFormValidation();
    
    // Auto-dismiss alerts
    autoDismissAlerts();
    
    // Intersection Observer for scroll animations
    initScrollAnimations();
    
    // Animate counters
    animateCounters();
});

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => {
        new bootstrap.Tooltip(el, {
            placement: 'top',
            trigger: 'hover'
        });
    });
}

/**
 * Form validation with visual feedback
 */
function initFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', (e) => {
            const submitBtn = form.querySelector('[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
                submitBtn.disabled = true;
            }
        });
    });
    
    // Password match validation for signup
    const confirmPassword = document.getElementById('confirm_password');
    const password = document.getElementById('password');
    
    if (confirmPassword && password) {
        confirmPassword.addEventListener('input', () => {
            if (confirmPassword.value !== password.value) {
                confirmPassword.setCustomValidity('Passwords do not match');
                confirmPassword.style.borderColor = '#f5576c';
            } else {
                confirmPassword.setCustomValidity('');
                confirmPassword.style.borderColor = '#43e97b';
            }
        });
    }
}

/**
 * Auto-dismiss flash alerts after 5 seconds
 */
function autoDismissAlerts() {
    const alerts = document.querySelectorAll('.glass-alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 5000);
    });
}

/**
 * Intersection Observer for scroll-triggered animations
 */
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    document.querySelectorAll('.animate-slide-up').forEach(el => {
        observer.observe(el);
    });
}

/**
 * Animate stat counter values
 */
function animateCounters() {
    const counters = document.querySelectorAll('.stat-value');
    counters.forEach(counter => {
        const text = counter.textContent.trim();
        const match = text.match(/^([\d.]+)(%?)$/);
        
        if (match) {
            const target = parseFloat(match[1]);
            const suffix = match[2] || '';
            const duration = 1500;
            const start = performance.now();
            
            counter.textContent = '0' + suffix;
            
            function step(timestamp) {
                const progress = Math.min((timestamp - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3); // Ease-out cubic
                const current = (target * eased).toFixed(target % 1 === 0 ? 0 : 2);
                counter.textContent = current + suffix;
                
                if (progress < 1) {
                    requestAnimationFrame(step);
                }
            }
            
            requestAnimationFrame(step);
        }
    });
}
