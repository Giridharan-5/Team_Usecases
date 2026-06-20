from src.agent_orch.agents.base_agent import BaseAgent
from src.agent_orch.utils import DBConnect

class RightsizingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RightsizingAgent")
        self.db = DBConnect.DBConnect()

    def _get_historical_metrics(self, server_id: str, resource_name: str) -> dict:
        """Get historical usage metrics (avg and max) from the last 30 records for the VM."""
        query = f"""
        SELECT 
            vpu_id,
            ROUND(AVG(avg_usage), 2) AS avg_usage,
            ROUND(AVG(max_usage), 2) AS max_usage
        FROM (
            SELECT 
                vu.vpu_id,
                vu.avg_usage,
                vu.max_usage,
                vu.dou
            FROM vm_usage_details vu
            INNER JOIN vm_primary_uri vp ON vp.vpu_id = vu.vpu_id
            INNER JOIN vm_details v ON v.vm_id = vp.vm_id
            WHERE v.vm_id = '{server_id}'
              AND vp.resource_type = '{resource_name}'
            ORDER BY vu.dou DESC
            LIMIT 30
        ) AS last_30_records
        GROUP BY vpu_id;
        """
        results = self.db.query(raw_query=query)
        return results[0] if results else None

    def _get_forecast_metrics(self, forecast_run_id: int) -> dict:
        """Get forecasted maximum usage from forecast results."""
        query = f"""
        SELECT 
            fru.vpu_id,
            ROUND(MAX(fr.p_avg_usage), 2) AS forecasted_max
        FROM forecast_results fr
        INNER JOIN forecast_runs fru ON fru.run_id = fr.run_id
        WHERE fr.run_id = '{forecast_run_id}'
        GROUP BY fru.vpu_id;
        """
        print(f"[RightsizingAgent] Forecast query for run_id={forecast_run_id},{query}")
        results = self.db.query(raw_query=query)
        print("[DEBUG] raw query result:", repr(results))
        if not results:
            print(f"[RightsizingAgent] No forecast data found for run_id={forecast_run_id}")
        return results[0] if results else None

    def _calculate_decision(self, forecasted_max: float, max_usage: float, avg_usage: float) -> tuple:
        """
        Calculate scaling decision and percentage based on forecast and historical metrics.
        
        Returns:
            tuple: (decision, scale_percent)
        """
        # Scale down conditions
        if forecasted_max < 35 and max_usage < 30 and avg_usage < 20:
            return 'scale_down', 50
        elif forecasted_max < 40 and max_usage < 45 and avg_usage < 30:
            return 'scale_down', 50
        elif forecasted_max < 55 and max_usage < 60 and avg_usage < 40:
            return 'scale_down', 25
        
        # Scale up conditions
        elif forecasted_max > 95 and avg_usage > 80:
            return 'scale_up', 50
        elif forecasted_max > 90 and avg_usage > 75:
            return 'scale_up', 25
        elif forecasted_max > 90 and avg_usage > 70:
            return 'scale_up', 25
        elif forecasted_max > 85 and avg_usage > 70:
            return 'scale_up', 25
        
        # No change
        else:
            return 'no_change', 0
        
    async def run(self, state: dict) -> dict:
        """
        Main execution method that orchestrates the rightsizing analysis.
        
        Steps:
        1. Get historical metrics (avg_usage, max_usage)
        2. Get forecast metrics (forecasted_max)
        3. Calculate decision using Python logic
        """
        server_id = state["server_id"]
        resource_name = state["resource_type"]
        forecast_run_id = state.get("run_id")
        if not forecast_run_id:
            warning = "Forecasting step did not produce a run_id; recommendation defaults to no forecast data."
            self.logger.warning(warning)
            warnings = state.get("warnings")
            if isinstance(warnings, list):
                warnings.append(warning)
            elif warnings:
                state["warnings"] = [warnings, warning]
            else:
                state["warnings"] = [warning]
            self.attach_result(state, key="recommendation", value={"decision": "no_forecast_data"})
            return state

        try:
            # Step 1: Get historical metrics
            hist_metrics = self._get_historical_metrics(server_id, resource_name)
            if not hist_metrics:
                self.attach_result(state, key="recommendation", value={"decision": "no_historical_data"})
                return state

            vpu_id = hist_metrics.get("vpu_id")
            avg_usage = hist_metrics.get("avg_usage", 0)
            max_usage = hist_metrics.get("max_usage", 0)

            # Step 2: Get forecast metrics
            forecast_metrics = self._get_forecast_metrics(forecast_run_id)
            if not forecast_metrics:
                self.attach_result(state, key="recommendation", value={"decision": "no_forecast_data"})
                return state

            forecasted_max = forecast_metrics.get("forecasted_max", 0)

            # Step 3: Calculate decision using Python logic
            decision, scale_percent = self._calculate_decision(forecasted_max, max_usage, avg_usage)

            # Build recommendation
            recommendation = {
                "vpu_id": vpu_id,
                "avg_usage": avg_usage,
                "max_usage": max_usage,
                "forecasted_max": forecasted_max,
                "decision": decision,
                "scale_percent": scale_percent
            }

            self.attach_result(state, key="recommendation", value=recommendation)
            print(f"[RightsizingAgent] Recommendation: {recommendation}")

        except Exception as e:
            print(f"[RightsizingAgent] Error: {e}")
            self.attach_result(state, key="recommendation", value={"error": str(e)})

        return state
    
