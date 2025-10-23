"""
Reviewer Agent - Meta-analysis of judge performance and system calibration.

The Reviewer periodically analyzes:
- Judge prediction accuracy and consistency
- Critic weight distributions (alphas)
- Overall system calibration quality
- Suggests adjustments to prompts or calibrator parameters
- Automatically improves underperforming judges
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import statistics
import datetime


@dataclass
class JudgeStats:
    """Statistics for a single judge."""
    judge_id: str
    predictions: List[float]  # r_tilde values
    errors: List[float]  # abs(r_tilde - true_rating)
    avg_error: float
    std_error: float
    consistency: float  # Lower is more consistent
    
    def __repr__(self) -> str:
        return (
            f"JudgeStats({self.judge_id}: "
            f"avg_err={self.avg_error:.3f}, "
            f"std={self.std_error:.3f}, "
            f"consistency={self.consistency:.3f})"
        )


@dataclass
class ReviewReport:
    """Comprehensive review of system performance."""
    timestamp: str
    num_predictions: int
    judge_stats: List[JudgeStats]
    overall_avg_error: float
    overall_std_error: float
    best_judge: Optional[str]
    worst_judge: Optional[str]
    recommendations: List[str]
    critic_utilization: Dict[str, float]  # How often each critic is weighted heavily
    
    def __repr__(self) -> str:
        return (
            f"ReviewReport({self.num_predictions} predictions, "
            f"avg_error={self.overall_avg_error:.3f}, "
            f"best={self.best_judge}, worst={self.worst_judge})"
        )


class Reviewer:
    """
    Meta-learning agent that analyzes judge performance and suggests improvements.
    
    The Reviewer:
    1. Collects statistics on judge predictions vs ground truth
    2. Analyzes critic weight distributions (alpha values)
    3. Identifies well-performing vs struggling judges
    4. Automatically improves underperforming judge prompts
    5. Creates evolved versions with better instructions
    """
    
    def __init__(self, review_interval: int = 5, resources_dir: Optional[str] = None, llm=None):
        """
        Initialize Reviewer.
        
        Args:
            review_interval: How often (in predictions) to run a review
            resources_dir: Path to resources directory for judge prompts
            llm: LLM client for generating improved prompts
        """
        self.review_interval = review_interval
        self.prediction_count = 0
        self.resources_dir = resources_dir
        self.llm = llm
        
        # Storage for analysis
        self.history: List[Dict[str, Any]] = []
        self.judge_predictions: Dict[str, List[float]] = {}
        self.judge_errors: Dict[str, List[float]] = {}
        self.critic_alpha_totals: Dict[str, List[float]] = {}
        
        # Track judge improvements
        self.judge_version_counter: Dict[str, int] = {}
        self.improvement_history: List[Dict[str, Any]] = []
        
        # Review reports
        self.reports: List[ReviewReport] = []
    
    def record_prediction(
        self,
        judge_outputs: List[Dict[str, Any]],
        true_rating: Optional[float] = None,
        critic_ids: Optional[List[str]] = None
    ) -> None:
        """
        Record a prediction for later analysis.
        
        Args:
            judge_outputs: List of judge output dictionaries
            true_rating: Ground truth rating (if available)
            critic_ids: List of critic IDs for alpha mapping
        """
        self.prediction_count += 1
        
        # Store raw data
        self.history.append({
            'judge_outputs': judge_outputs,
            'true_rating': true_rating,
            'critic_ids': critic_ids,
            'prediction_num': self.prediction_count
        })
        
        # Process judge outputs
        for j_out in judge_outputs:
            judge_id = j_out.get('judge_id', 'unknown')
            r_tilde = float(j_out.get('r_tilde', 0.0))
            
            # Track predictions
            if judge_id not in self.judge_predictions:
                self.judge_predictions[judge_id] = []
            self.judge_predictions[judge_id].append(r_tilde)
            
            # Track errors if ground truth available
            if true_rating is not None:
                error = abs(r_tilde - true_rating)
                if judge_id not in self.judge_errors:
                    self.judge_errors[judge_id] = []
                self.judge_errors[judge_id].append(error)
            
            # Track critic utilization (alpha weights)
            alphas = j_out.get('alphas', [])
            if alphas and critic_ids and len(alphas) == len(critic_ids):
                for cid, alpha in zip(critic_ids, alphas):
                    if cid not in self.critic_alpha_totals:
                        self.critic_alpha_totals[cid] = []
                    try:
                        self.critic_alpha_totals[cid].append(float(alpha))
                    except (ValueError, TypeError):
                        pass
    
    def should_review(self) -> bool:
        """Check if it's time to run a review."""
        return self.prediction_count > 0 and self.prediction_count % self.review_interval == 0
    
    def run_review(self) -> ReviewReport:
        """
        Analyze accumulated data and generate a review report.
        
        Returns:
            ReviewReport with statistics and recommendations
        """
        import datetime
        
        # Calculate judge statistics
        judge_stats_list = []
        for judge_id, predictions in self.judge_predictions.items():
            errors = self.judge_errors.get(judge_id, [])
            
            if errors:
                avg_error = statistics.mean(errors)
                std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
                consistency = std_error  # Lower std = more consistent
            else:
                avg_error = 0.0
                std_error = 0.0
                consistency = 0.0
            
            judge_stats_list.append(JudgeStats(
                judge_id=judge_id,
                predictions=predictions.copy(),
                errors=errors.copy(),
                avg_error=avg_error,
                std_error=std_error,
                consistency=consistency
            ))
        
        # Overall statistics
        all_errors = []
        for errors in self.judge_errors.values():
            all_errors.extend(errors)
        
        overall_avg_error = statistics.mean(all_errors) if all_errors else 0.0
        overall_std_error = statistics.stdev(all_errors) if len(all_errors) > 1 else 0.0
        
        # Find best/worst judges
        best_judge = None
        worst_judge = None
        if judge_stats_list:
            sorted_by_error = sorted(judge_stats_list, key=lambda x: x.avg_error)
            best_judge = sorted_by_error[0].judge_id
            worst_judge = sorted_by_error[-1].judge_id
        
        # Critic utilization
        critic_util = {}
        for cid, alphas in self.critic_alpha_totals.items():
            if alphas:
                critic_util[cid] = statistics.mean(alphas)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            judge_stats_list, critic_util, overall_avg_error
        )
        
        # Create report
        report = ReviewReport(
            timestamp=datetime.datetime.now().isoformat(),
            num_predictions=self.prediction_count,
            judge_stats=judge_stats_list,
            overall_avg_error=overall_avg_error,
            overall_std_error=overall_std_error,
            best_judge=best_judge,
            worst_judge=worst_judge,
            recommendations=recommendations,
            critic_utilization=critic_util
        )
        
        self.reports.append(report)
        return report
    
    def _generate_recommendations(
        self,
        judge_stats: List[JudgeStats],
        critic_util: Dict[str, float],
        overall_error: float
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            judge_stats: Statistics for each judge
            critic_util: Average alpha weights per critic
            overall_error: Overall system error
        
        Returns:
            List of recommendation strings
        """
        recs = []
        
        # Check overall performance
        if overall_error > 1.5:
            recs.append("⚠️  Overall error is high (>1.5). Consider retraining calibrator or adjusting judge prompts.")
        elif overall_error < 0.5:
            recs.append("✅ Excellent overall performance! System is well-calibrated.")
        
        # Check judge consistency
        if judge_stats:
            inconsistent_judges = [
                js for js in judge_stats 
                if js.std_error > 1.0 and len(js.errors) > 2
            ]
            if inconsistent_judges:
                recs.append(
                    f"📊 Inconsistent judges detected: {', '.join(j.judge_id for j in inconsistent_judges)}. "
                    "Consider refining their prompts for more stable outputs."
                )
            
            # Check for underperforming judges
            if len(judge_stats) > 1:
                avg_judge_error = statistics.mean([js.avg_error for js in judge_stats])
                underperforming = [
                    js for js in judge_stats
                    if js.avg_error > avg_judge_error * 1.3
                ]
                if underperforming:
                    recs.append(
                        f"⚡ Underperforming judges: {', '.join(j.judge_id for j in underperforming)}. "
                        "Review their prompts or increase calibrator dimension."
                    )
        
        # Check critic utilization
        if critic_util:
            underutilized = [cid for cid, avg_alpha in critic_util.items() if avg_alpha < 0.1]
            overutilized = [cid for cid, avg_alpha in critic_util.items() if avg_alpha > 0.5]
            
            if underutilized:
                recs.append(
                    f"🔍 Underutilized critics: {', '.join(underutilized)}. "
                    "These critics may need prompt improvements or are rarely relevant."
                )
            
            if overutilized:
                recs.append(
                    f"⭐ Heavily weighted critics: {', '.join(overutilized)}. "
                    "These critics are highly trusted by judges."
                )
        
        # Calibrator suggestions
        if overall_error > 1.0:
            recs.append(
                "🔧 Consider increasing calibrator_dim in OrchestratorConfig for more expressive calibration."
            )
        
        if not recs:
            recs.append("👍 No major issues detected. System is performing well!")
        
        return recs
    
    def print_report(self, report: ReviewReport) -> None:
        """
        Pretty-print a review report.
        
        Args:
            report: ReviewReport to display
        """
        print(f"\n{'='*80}")
        print(f"🔍 REVIEWER ANALYSIS - {report.timestamp}")
        print(f"{'='*80}")
        print(f"Total predictions analyzed: {report.num_predictions}")
        print(f"Overall avg error: {report.overall_avg_error:.3f} ± {report.overall_std_error:.3f}")
        print()
        
        # Judge performance
        if report.judge_stats:
            print("📊 Judge Performance:")
            for js in sorted(report.judge_stats, key=lambda x: x.avg_error):
                print(f"  • {js.judge_id:20s} | "
                      f"Avg Error: {js.avg_error:.3f} | "
                      f"Std: {js.std_error:.3f} | "
                      f"Predictions: {len(js.predictions)}")
            print()
            
            if report.best_judge:
                print(f"🏆 Best performing judge: {report.best_judge}")
            if report.worst_judge:
                print(f"⚠️  Needs improvement: {report.worst_judge}")
            print()
        
        # Critic utilization
        if report.critic_utilization:
            print("🎭 Critic Utilization (avg alpha weights):")
            sorted_critics = sorted(
                report.critic_utilization.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for cid, avg_alpha in sorted_critics:
                bar_len = int(avg_alpha * 20)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                print(f"  • {cid:20s} [{bar}] {avg_alpha:.3f}")
            print()
        
        # Recommendations
        if report.recommendations:
            print("💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  {rec}")
        
        print(f"{'='*80}\n")
    
    def suggest_calibrator_update(self) -> Optional[Dict[str, Any]]:
        """
        Suggest calibrator parameter updates based on analysis.
        
        Returns:
            Dictionary with suggested calibrator updates, or None
        """
        if not self.reports:
            return None
        
        latest_report = self.reports[-1]
        suggestions = {}
        
        # Suggest dimension increase if error is high
        if latest_report.overall_avg_error > 1.2:
            suggestions['increase_dim'] = True
            suggestions['reason'] = "High error suggests underfitting"
        
        # Suggest learning rate adjustment based on error trend
        if len(self.reports) > 1:
            prev_error = self.reports[-2].overall_avg_error
            curr_error = latest_report.overall_avg_error
            
            if curr_error > prev_error * 1.1:
                suggestions['decrease_learning_rate'] = True
                suggestions['reason'] = "Error increasing - may be overfitting"
        
        return suggestions if suggestions else None
    
    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self.history.clear()
        self.judge_predictions.clear()
        self.judge_errors.clear()
        self.critic_alpha_totals.clear()
        self.prediction_count = 0
    
    def improve_worst_judge(self, report: ReviewReport) -> Optional[Dict[str, Any]]:
        """
        Automatically improve the worst performing judge by:
        1. Reading its current prompt
        2. Analyzing its errors
        3. Generating an improved version with LLM
        4. Creating a new judge file with versioned name
        
        Args:
            report: ReviewReport with judge statistics
            
        Returns:
            Dict with improvement details or None if cannot improve
        """
        if not report.worst_judge or not self.resources_dir or not self.llm:
            return None
        
        worst_judge_id = report.worst_judge
        worst_stats = next((js for js in report.judge_stats if js.judge_id == worst_judge_id), None)
        
        if not worst_stats or worst_stats.avg_error < 0.7:
            # Don't improve if error is already low
            return None
        
        # Read current prompt
        judges_dir = Path(self.resources_dir) / "judges"
        current_prompt_path = judges_dir / f"{worst_judge_id}.txt"
        
        if not current_prompt_path.exists():
            print(f"⚠️  Cannot find prompt file: {current_prompt_path}")
            return None
        
        with open(current_prompt_path, 'r') as f:
            current_prompt = f.read()
        
        # Increment version counter
        if worst_judge_id not in self.judge_version_counter:
            self.judge_version_counter[worst_judge_id] = 1
        else:
            self.judge_version_counter[worst_judge_id] += 1
        
        version = self.judge_version_counter[worst_judge_id]
        new_judge_id = f"{worst_judge_id}_v{version}"
        
        # Generate improvement prompt for LLM
        improvement_request = f"""You are an expert at improving judge prompts for movie rating prediction systems.

CURRENT JUDGE: {worst_judge_id}
PERFORMANCE STATS:
- Average Error: {worst_stats.avg_error:.3f}
- Consistency (std): {worst_stats.std_error:.3f}
- Number of predictions: {len(worst_stats.predictions)}

CURRENT PROMPT:
{current_prompt}

TASK:
Improve this judge prompt to reduce prediction errors. The judge's role is to:
1. Evaluate multiple critic opinions about a movie
2. Assign weights (alphas) to each critic based on their reasoning quality
3. Produce a calibrated rating prediction (r_tilde)

SPECIFIC IMPROVEMENTS NEEDED:
- Better instructions for handling conflicting critic opinions
- More precise guidance on assigning alpha weights
- Clearer criteria for evaluating critic reasoning quality
- Better calibration of final prediction based on weighted evidence

Return ONLY the improved prompt text, no explanations or metadata.
The prompt should be concise but comprehensive (300-500 words).
"""

        try:
            print(f"\n🔧 Generating improved prompt for {worst_judge_id}...")
            improved_prompt = self.llm.generate(
                system_prompt="""You are proposing improvements to a judge prompt where you have to varied the personality and the tone of the judge. EXAMPLE 
                You are a confidence-weighted judge who trusts self-assured critics with strong convictions.
Amplify the influence of critics expressing high confidence. Discount uncertain or hedging opinions.
MUST HAVE THE FOLLOWING INSTRUCTIONS:
Verify each critic rationale against FACTS. Set flags=1 for unsupported claims.
Weight critics with supported rationales and reliable confidence.
Produce normalized alphas weighing the relevance of the opinion of each critic, and report the corresponding r_tilde such that it is a weighted average of the critic scores and the alphas you created.
Output STRICT JSON only.

                """,
                user_prompt=improvement_request,
            )
            
            # Save new prompt
            new_prompt_path = judges_dir / f"{new_judge_id}.txt"
            with open(new_prompt_path, 'w') as f:
                f.write(improved_prompt)
            
            # Log improvement
            improvement_info = {
                'timestamp': datetime.datetime.now().isoformat(),
                'original_judge': worst_judge_id,
                'new_judge': new_judge_id,
                'original_error': worst_stats.avg_error,
                'original_std': worst_stats.std_error,
                'version': version,
                'prompt_path': str(new_prompt_path),
                'reason': f"High error ({worst_stats.avg_error:.3f}) and/or inconsistency ({worst_stats.std_error:.3f})"
            }
            self.improvement_history.append(improvement_info)
            
            print(f"✅ Created improved judge: {new_judge_id}")
            print(f"📝 Saved to: {new_prompt_path}")
            print(f"📊 Original error: {worst_stats.avg_error:.3f} → Target: < 0.7")
            print(f"🎯 Why: {improvement_info['reason']}")
            
            # Show snippet of changes
            print(f"\n📄 New prompt preview:")
            preview = improved_prompt[:200].replace('\n', ' ')
            print(f"   {preview}...")
            
            return improvement_info
            
        except Exception as e:
            print(f"❌ Error generating improved prompt: {e}")
            return None
