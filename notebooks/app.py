if engagement_is_key:
                st.caption("Note: Engagement behavior is more influential than demographics (e.g., age, gender, location).")
            
            # Download predicted results
            st.subheader("Download Predictions")
            csv = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Please upload a CSV file from the sidebar to see predictions.")