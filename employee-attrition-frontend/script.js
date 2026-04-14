// Smooth scrolling functions
function scrollToDashboard() {
    document.getElementById('dashboard').scrollIntoView({ behavior: 'smooth' });
}

function scrollToPredict() {
    document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
}

// Active navigation highlight
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Range slider value displays
document.getElementById('jobSatisfaction').addEventListener('input', (e) => {
    document.getElementById('satisfactionValue').textContent = e.target.value;
});

document.getElementById('jobInvolvement').addEventListener('input', (e) => {
    document.getElementById('involvementValue').textContent = e.target.value;
});

document.getElementById('envSatisfaction').addEventListener('input', (e) => {
    document.getElementById('envValue').textContent = e.target.value;
});

document.getElementById('stockOption').addEventListener('input', (e) => {
    document.getElementById('stockValue').textContent = e.target.value;
});

document.getElementById('workLifeBalance').addEventListener('input', (e) => {
    document.getElementById('balanceValue').textContent = e.target.value;
});

// Prediction logic based on SHAP values from the model
document.getElementById('predictionForm').addEventListener('submit', (e) => {
    e.preventDefault();
    
    // Get form values
    const age = parseInt(document.getElementById('age').value);
    const department = document.getElementById('department').value;
    const jobSatisfaction = parseInt(document.getElementById('jobSatisfaction').value);
    const jobInvolvement = parseInt(document.getElementById('jobInvolvement').value);
    const envSatisfaction = parseInt(document.getElementById('envSatisfaction').value);
    const stockOption = parseInt(document.getElementById('stockOption').value);
    const overtime = document.getElementById('overtime').value;
    const income = parseInt(document.getElementById('income').value);
    const workLifeBalance = parseInt(document.getElementById('workLifeBalance').value);
    const yearsAtCompany = parseInt(document.getElementById('yearsAtCompany').value);
    
    // Calculate risk score based on actual SHAP values from the model
    let riskScore = 0;
    const riskFactors = [];
    
    // Stock option impact (from SHAP: 0.0984 - most important factor)
    if (stockOption === 0) {
        riskScore += 25;
        riskFactors.push('No stock options (strongest predictor)');
    } else if (stockOption === 1) {
        riskScore += 15;
        riskFactors.push('Limited stock options');
    } else if (stockOption === 2) {
        riskScore += 5;
    }
    
    // Job satisfaction impact (from SHAP: 0.0546 - 2nd most important)
    if (jobSatisfaction <= 2) {
        riskScore += 20;
        riskFactors.push('Low job satisfaction');
    } else if (jobSatisfaction === 3) {
        riskScore += 10;
    }
    
    // Job involvement impact (from SHAP: 0.0459 - 3rd most important)
    if (jobInvolvement <= 2) {
        riskScore += 15;
        riskFactors.push('Low job involvement');
    }
    
    // Environment satisfaction impact (from SHAP: 0.0447 - 4th most important)
    if (envSatisfaction <= 2) {
        riskScore += 15;
        riskFactors.push('Poor work environment satisfaction');
    }
    
    // Overtime impact
    if (overtime === 'Yes') {
        riskScore += 20;
        riskFactors.push('Regular overtime work');
    }
    
    // Years at company (high risk in first 1-2 years)
    if (yearsAtCompany <= 2) {
        riskScore += 10;
        riskFactors.push('New employee (< 2 years tenure)');
    }
    
    // Work-life balance
    if (workLifeBalance <= 2) {
        riskScore += 10;
        riskFactors.push('Poor work-life balance');
    }
    
    // Age factor (younger employees at higher risk)
    if (age < 30) {
        riskScore += 5;
        riskFactors.push('Young employee (< 30 years)');
    }
    
    // Income factor (lower income increases risk)
    if (income < 3000) {
        riskScore += 5;
        riskFactors.push('Lower than average income');
    }
    
    // Department risk based on actual data
    if (department === 'Sales') {
        riskScore += 10;
        riskFactors.push('High-risk department (Sales)');
    } else if (department === 'Human Resources') {
        riskScore += 5;
        riskFactors.push('Medium-risk department (HR)');
    }
    
    // Cap at 100%
    riskScore = Math.min(riskScore, 100);
    
    // Get unique risk factors (first 4)
    const uniqueFactors = [...new Set(riskFactors)].slice(0, 4);
    
    // Display result
    const resultDiv = document.getElementById('predictResult');
    const riskPercentage = document.getElementById('riskPercentage');
    const riskFill = document.getElementById('riskFill');
    const riskStatus = document.getElementById('riskStatus');
    const riskRecommendation = document.getElementById('riskRecommendation');
    const riskFactorsList = document.getElementById('riskFactors');
    
    resultDiv.style.display = 'block';
    
    // Animate gauge
    setTimeout(() => {
        const roundedScore = Math.round(riskScore);
        riskPercentage.textContent = `${roundedScore}%`;
        riskFill.style.width = `${riskScore}%`;
        
        // Set color and message based on risk level
        if (riskScore >= 70) {
            riskFill.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
            riskStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> HIGH RISK - Immediate Action Required';
            riskStatus.style.color = '#ef4444';
            riskRecommendation.innerHTML = '🚨 Schedule immediate 1:1 meeting, review compensation package, offer stock options, and create retention plan.';
        } else if (riskScore >= 40) {
            riskFill.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
            riskStatus.innerHTML = '<i class="fas fa-chart-line"></i> MEDIUM RISK - Monitor Closely';
            riskStatus.style.color = '#f59e0b';
            riskRecommendation.innerHTML = '📊 Regular check-ins, career development discussions, engagement activities, and consider stock options.';
        } else {
            riskFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
            riskStatus.innerHTML = '<i class="fas fa-check-circle"></i> LOW RISK - Stable Employee';
            riskStatus.style.color = '#10b981';
            riskRecommendation.innerHTML = '✅ Continue engagement, provide recognition, offer growth opportunities, and maintain work-life balance.';
        }
        
        // Update risk factors list
        riskFactorsList.innerHTML = '';
        if (uniqueFactors.length > 0) {
            uniqueFactors.forEach(factor => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fas fa-exclamation-circle" style="color: #ef4444; margin-right: 8px;"></i>${factor}`;
                riskFactorsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.innerHTML = '<i class="fas fa-check-circle" style="color: #10b981; margin-right: 8px;"></i>No major risk factors detected';
            riskFactorsList.appendChild(li);
        }
        
        // Scroll to result
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
});

// Add loading state for images
window.addEventListener('load', () => {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('error', () => {
            console.log(`Image not found: ${img.src}`);
            // Optional: Show a fallback message
            const parent = img.parentElement;
            if (parent && !parent.querySelector('.error-message')) {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.style.padding = '20px';
                errorMsg.style.textAlign = 'center';
                errorMsg.style.color = '#ef4444';
                errorMsg.innerHTML = '<i class="fas fa-image"></i> Image not found<br><small>Run the Python script to generate this visualization</small>';
                parent.appendChild(errorMsg);
            }
        });
    });
    
    console.log('Frontend loaded successfully!');
});

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    .gauge-fill {
        transition: width 0.5s ease-out;
    }
    
    .risk-status i, .risk-recommendation i {
        margin-right: 8px;
    }
    
    .risk-factors ul {
        list-style: none;
        padding: 0;
    }
    
    .risk-factors li {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .error-message {
        font-size: 0.875rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .dashboard-card, .insight-card-large, .predict-form, .predict-result {
        animation: fadeIn 0.6s ease-out;
    }
`;
document.head.appendChild(style);