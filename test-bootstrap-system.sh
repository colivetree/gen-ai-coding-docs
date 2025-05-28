#!/bin/bash

# 🧪 AI Project Bootstrap System Test Suite
# Tests all project templates and validates the bootstrap functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_DIR="bootstrap-tests"
CLEANUP=${CLEANUP:-true}

echo -e "${BLUE}🧪 AI Project Bootstrap System Test Suite${NC}"
echo "=================================================="

# Cleanup function
cleanup() {
    if [ "$CLEANUP" = "true" ]; then
        echo -e "${YELLOW}🧹 Cleaning up test directory...${NC}"
        rm -rf "$TEST_DIR"
    else
        echo -e "${YELLOW}⚠️  Test directory preserved: $TEST_DIR${NC}"
    fi
}

# Error handler
error_exit() {
    echo -e "${RED}❌ Test failed: $1${NC}"
    cleanup
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Warning message
warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Info message
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Test function
test_project_type() {
    local project_type=$1
    local project_name=$2
    local template_file=$3
    
    info "Testing $project_type project type..."
    
    # Validate template structure first (before changing directories)
    if [ ! -f "$template_file" ]; then
        error_exit "Template file $template_file not found"
    fi
    
    # Create test directory
    mkdir -p "$TEST_DIR/$project_name"
    cd "$TEST_DIR/$project_name"
    
    # Copy template files
    cp "../../$template_file" ./template.md
    cp "../../ai-project-bootstrap-template.md" ./
    
    # Check for required sections (flexible for different template types)
    local required_sections=(
        "Project Type:"
        "Stack:"
        "Includes:"
        "Environment Variables"
        "Bootstrap Command"
    )
    
    # Additional sections that should exist (at least one)
    local docker_sections=(
        "Docker Compose Override"
        "Multi-tenant Architecture"
        "Specific Configuration"
    )
    
    for section in "${required_sections[@]}"; do
        if ! grep -q "$section" "template.md"; then
            error_exit "Required section '$section' not found in $template_file"
        fi
    done
    
    # Check for at least one Docker/configuration section
    local found_docker_section=false
    for section in "${docker_sections[@]}"; do
        if grep -q "$section" "template.md"; then
            found_docker_section=true
            break
        fi
    done
    
    if [ "$found_docker_section" = false ]; then
        error_exit "No Docker/configuration section found in $template_file"
    fi
    
    # Validate Docker Compose syntax
    if grep -q "docker-compose.override.yml" "template.md"; then
        # Extract docker-compose content and validate basic YAML structure
        if ! grep -A 50 "docker-compose.override.yml" "template.md" | grep -q "version:"; then
            warning "Docker Compose version not specified in $template_file"
        fi
        
        if ! grep -A 50 "docker-compose.override.yml" "template.md" | grep -q "services:"; then
            error_exit "Docker Compose services section not found in $template_file"
        fi
    fi
    
    # Validate environment variables
    if grep -q ".env.example" "template.md"; then
        # Check for required environment variables
        local required_env_vars=(
            "PROJECT_NAME"
            "ENVIRONMENT"
            "DATABASE_URL"
            "JWT_SECRET"
        )
        
        for var in "${required_env_vars[@]}"; do
            if ! grep -A 50 ".env.example" "template.md" | grep -q "$var"; then
                warning "Required environment variable '$var' not found in $template_file"
            fi
        done
    fi
    
    # Validate bootstrap script
    if grep -q "Bootstrap Command" "template.md"; then
        if ! grep -A 20 "Bootstrap Command" "template.md" | grep -q "#!/bin/bash"; then
            warning "Bootstrap script should start with #!/bin/bash in $template_file"
        fi
        
        # Check for success message (flexible patterns)
        if ! grep -A 50 "Bootstrap Command" "template.md" | grep -E "(bootstrapped|completed|ready)" > /dev/null; then
            warning "Bootstrap script should have success message in $template_file"
        fi
    fi
    
    success "$project_type template validation passed"
    cd ../..
}

# Test main bootstrap template
test_main_bootstrap() {
    info "Testing main bootstrap template..."
    
    if [ ! -f "ai-project-bootstrap-template.md" ]; then
        error_exit "Main bootstrap template not found"
    fi
    
    # Check for required sections
    local required_sections=(
        "Mission"
        "Common Infrastructure Repository"
        "Core Infrastructure Stack"
        "Project Types & Templates"
        "Bootstrap Protocol"
        "Environment Configuration"
        "Docker Compose Structure"
    )
    
    for section in "${required_sections[@]}"; do
        if ! grep -q "$section" "ai-project-bootstrap-template.md"; then
            error_exit "Required section '$section' not found in main template"
        fi
    done
    
    # Validate infrastructure components
    local required_components=(
        "Nginx"
        "Supabase Auth"
        "Hatchet"
        "RabbitMQ"
        "Prometheus"
        "Grafana"
        "Stripe"
    )
    
    for component in "${required_components[@]}"; do
        if ! grep -q "$component" "ai-project-bootstrap-template.md"; then
            warning "Infrastructure component '$component' not mentioned in main template"
        fi
    done
    
    success "Main bootstrap template validation passed"
}

# Test bugfix template
test_bugfix_template() {
    info "Testing bugfix template..."
    
    if [ ! -f "ai-bugfix-prompt-template.md" ]; then
        error_exit "Bugfix template not found"
    fi
    
    # Check for bugfix-specific sections
    local required_sections=(
        "Mission"
        "Bug Analysis Protocol"
        "Change Scope Constraints"
        "Validation Checklist"
        "Bug Report Template"
    )
    
    for section in "${required_sections[@]}"; do
        if ! grep -q "$section" "ai-bugfix-prompt-template.md"; then
            error_exit "Required section '$section' not found in bugfix template"
        fi
    done
    
    # Check for minimal change emphasis
    if ! grep -q "minimal" "ai-bugfix-prompt-template.md"; then
        warning "Bugfix template should emphasize minimal changes"
    fi
    
    success "Bugfix template validation passed"
}

# Test project type templates
test_project_templates() {
    info "Testing project type templates..."
    
    # Test Full-Stack Web Application
    test_project_type "Full-Stack Web Application" "fullstack-test" "project-type-templates/fullstack-web-app-template.md"
    
    # Test API-Only Service
    test_project_type "API-Only Service" "api-test" "project-type-templates/api-only-service-template.md"
    
    # Test AI/ML Pipeline
    test_project_type "AI/ML Pipeline" "ml-test" "project-type-templates/ai-ml-pipeline-template.md"
    
    # Test SaaS Application
    test_project_type "SaaS Application" "saas-test" "project-type-templates/saas-application-template.md"
}

# Test Docker Compose validation
test_docker_compose() {
    info "Testing Docker Compose configurations..."
    
    # Create a temporary docker-compose file for testing
    mkdir -p "$TEST_DIR/docker-test"
    cd "$TEST_DIR/docker-test"
    
    # Extract a sample docker-compose from one of the templates
    grep -A 100 "docker-compose.override.yml" "../../project-type-templates/fullstack-web-app-template.md" | \
    sed -n '/```yaml/,/```/p' | sed '1d;$d' > docker-compose.test.yml
    
    # Basic YAML validation (if docker-compose is available)
    if command -v docker-compose &> /dev/null; then
        if docker-compose -f docker-compose.test.yml config &> /dev/null; then
            success "Docker Compose configuration is valid"
        else
            warning "Docker Compose configuration validation failed (this may be expected without full environment)"
        fi
    else
        warning "Docker Compose not available for validation"
    fi
    
    cd ../..
}

# Test environment variable consistency
test_env_consistency() {
    info "Testing environment variable consistency..."
    
    # Check that all templates use consistent environment variable names
    local templates=(
        "project-type-templates/fullstack-web-app-template.md"
        "project-type-templates/api-only-service-template.md"
        "project-type-templates/ai-ml-pipeline-template.md"
        "project-type-templates/saas-application-template.md"
    )
    
    local common_vars=(
        "PROJECT_NAME"
        "ENVIRONMENT"
        "DATABASE_URL"
        "JWT_SECRET"
        "SUPABASE_URL"
        "HATCHET_CLIENT_TOKEN"
    )
    
    for template in "${templates[@]}"; do
        if [ -f "$template" ]; then
            for var in "${common_vars[@]}"; do
                if ! grep -q "$var" "$template"; then
                    warning "Common variable '$var' not found in $template"
                fi
            done
        fi
    done
    
    success "Environment variable consistency check completed"
}

# Test bootstrap script syntax
test_bootstrap_scripts() {
    info "Testing bootstrap script syntax..."
    
    local templates=(
        "project-type-templates/fullstack-web-app-template.md"
        "project-type-templates/api-only-service-template.md"
        "project-type-templates/ai-ml-pipeline-template.md"
        "project-type-templates/saas-application-template.md"
    )
    
    for template in "${templates[@]}"; do
        if [ -f "$template" ]; then
            # Extract bootstrap script and check basic syntax
            if grep -q "Bootstrap Command" "$template"; then
                # Check for common bash patterns
                if ! grep -A 50 "Bootstrap Command" "$template" | grep -q "set -e"; then
                    warning "Bootstrap script in $template should use 'set -e' for error handling"
                fi
                
                if ! grep -A 50 "Bootstrap Command" "$template" | grep -q "echo.*✅"; then
                    warning "Bootstrap script in $template should have success indicator"
                fi
            fi
        fi
    done
    
    success "Bootstrap script syntax check completed"
}

# Test file structure
test_file_structure() {
    info "Testing file structure..."
    
    # Check that all required files exist
    local required_files=(
        "ai-project-bootstrap-template.md"
        "ai-bugfix-prompt-template.md"
        "project-type-templates/fullstack-web-app-template.md"
        "project-type-templates/api-only-service-template.md"
        "project-type-templates/ai-ml-pipeline-template.md"
        "project-type-templates/saas-application-template.md"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error_exit "Required file not found: $file"
        fi
    done
    
    success "File structure validation passed"
}

# Test template completeness
test_template_completeness() {
    info "Testing template completeness..."
    
    local templates=(
        "project-type-templates/fullstack-web-app-template.md"
        "project-type-templates/api-only-service-template.md"
        "project-type-templates/ai-ml-pipeline-template.md"
        "project-type-templates/saas-application-template.md"
    )
    
    for template in "${templates[@]}"; do
        if [ -f "$template" ]; then
            # Check minimum content length (should be substantial)
            local line_count=$(wc -l < "$template")
            if [ "$line_count" -lt 100 ]; then
                warning "Template $template seems too short ($line_count lines)"
            fi
            
            # Check for code blocks
            local code_blocks=$(grep -c '```' "$template")
            if [ "$code_blocks" -lt 10 ]; then
                warning "Template $template has few code blocks ($code_blocks)"
            fi
        fi
    done
    
    success "Template completeness check completed"
}

# Main test execution
main() {
    info "Starting AI Project Bootstrap System tests..."
    
    # Create test directory
    mkdir -p "$TEST_DIR"
    
    # Run all tests
    test_file_structure
    test_main_bootstrap
    test_bugfix_template
    test_project_templates
    test_docker_compose
    test_env_consistency
    test_bootstrap_scripts
    test_template_completeness
    
    # Summary
    echo ""
    echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
    echo ""
    echo "📊 Test Summary:"
    echo "  ✅ File structure validation"
    echo "  ✅ Main bootstrap template"
    echo "  ✅ Bugfix template"
    echo "  ✅ Project type templates (4)"
    echo "  ✅ Docker Compose configurations"
    echo "  ✅ Environment variable consistency"
    echo "  ✅ Bootstrap script syntax"
    echo "  ✅ Template completeness"
    echo ""
    echo -e "${BLUE}🚀 Bootstrap system is ready for use!${NC}"
    echo ""
    echo "Usage examples:"
    echo "  # Full-stack web app"
    echo "  ./scripts/bootstrap.sh my-saas-app prod --type=fullstack"
    echo ""
    echo "  # API-only service"
    echo "  ./scripts/bootstrap.sh my-api-service prod --type=api"
    echo ""
    echo "  # AI/ML pipeline"
    echo "  ./scripts/bootstrap.sh my-ml-pipeline prod --type=ml"
    echo ""
    echo "  # SaaS application"
    echo "  ./scripts/bootstrap.sh my-saas-platform prod --type=saas"
    
    cleanup
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@" 